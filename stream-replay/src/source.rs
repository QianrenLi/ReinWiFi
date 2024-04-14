use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};
use ndarray::prelude::*;
use ndarray_npy::read_npy;

use crate::conf::{StreamParam, ConnParams};
use crate::packet::*;
use crate::broker::GlobalBroker;
use crate::throttle::RateThrottler;
use crate::rtt::{RttRecorder,RttSender};
use crate::dispatcher::BlockedSignal;
use crate::ipc::Statistics;

type GuardedThrottler = Arc<Mutex<RateThrottler>>;

pub fn source_thread(throttler:GuardedThrottler, rtt_tx: Option<RttSender>,
    params: ConnParams, tx:PacketSender, blocked_signal:BlockedSignal)
{
    let trace: Array2<u64> = read_npy(&params.npy_file).expect("loading failed.");
    let (start_offset, duration) = (params.start_offset, params.duration);
    let mut template = PacketStruct::new(params.port);
    let spin_sleeper = spin_sleep::SpinSleeper::new(100_000)
                        .with_spin_strategy(spin_sleep::SpinStrategy::YieldThread);

    let mut loops = 0;
    let mut idx = start_offset;
    let stop_time  = SystemTime::now().checked_add( Duration::from_secs_f64(duration[1]) ).unwrap();

    spin_sleeper.sleep( Duration::from_secs_f64(duration[0]) );
    while SystemTime::now() <= stop_time {
        loops += 1;

        let deadline = if loops < params.loops {
            // 0. next iteration
            idx = (idx + 1) % trace.shape()[0];
            let size_bytes = trace[[idx, 1]] as usize;
            let interval_ns = trace[[idx, 0]];

            // 1. generate packets
            let mut packets = Vec::new();
            let (_num, _remains) = (size_bytes/UDP_MAX_LENGTH, size_bytes%UDP_MAX_LENGTH);
            template.next_seq(_num, _remains);
            template.set_length(UDP_MAX_LENGTH as u16);
            for _ in 0.._num {
                packets.push( template.clone() );
                template.next_offset();
            }
            if _remains > 0 {
                template.next_offset();
                template.set_length(_remains as u16);
                packets.push( template.clone() );
            }

            // 2. append to application-layer queue
            throttler.lock().unwrap().prepare( packets );
            // report RTT
            if let Some(ref r_tx) = rtt_tx {
                r_tx.send(template.seq).unwrap();
            }

            SystemTime::now() + Duration::from_nanos(interval_ns)
        }
        else {
            stop_time
        };
        
        // 3. process queue, aware of blocked status
        while SystemTime::now() < deadline {
            let _signal = blocked_signal.lock().unwrap();
            if !(*_signal) {
                match throttler.lock().unwrap().try_consume(|mut packet| {
                    let time_now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f64();
                    packet.timestamp = time_now;
                    tx.send(packet).unwrap();
                    true
                }) {
                    Some(_) => continue,
                    None=> break
                }
            }
        }

        // 4. sleep until next arrival
        if let Ok(remaining_time) = deadline.duration_since( SystemTime::now() ) {
            spin_sleeper.sleep( remaining_time );
        }
    }

    //reset throttler
    throttler.lock().unwrap().reset();
}

pub struct SourceManager{
    pub name: String,
    stream: StreamParam,
    //
    start_timestamp: SystemTime,
    stop_timestamp: SystemTime,
    //
    throttler: GuardedThrottler,
    rtt: Option<RttRecorder>,
    //
    tx: Vec<PacketSender>,
    blocked_signal: BlockedSignal
}

impl SourceManager {
    pub fn new(stream: StreamParam, window_size:usize, broker:&mut GlobalBroker) -> Self {
        let (StreamParam::UDP(ref params) | StreamParam::TCP(ref params)) = stream;
        let name = format!("{}@{}", params.port, params.tos);
        let (tx, blocked_signal) = broker.add(params.tos, params.priority.clone());
        let tx = [tx].into();

        let throttler = Arc::new(Mutex::new(
            RateThrottler::new(name.clone(), params.throttle, window_size, params.no_logging, params.loops != usize::MAX)
        ));
        let rtt =  match params.calc_rtt {
            false => None,
            true => Some( RttRecorder::new( &name, params.port ) )
        };

        let start_timestamp = SystemTime::now();
        let stop_timestamp = SystemTime::now();

        Self{ name, stream, throttler, rtt, tx, blocked_signal, start_timestamp, stop_timestamp }
    }

    pub fn throttle(&self, throttle:f64) {
        if let Ok(ref mut throttler) = self.throttler.lock() {
            throttler.throttle = throttle;
        }
    }

    pub fn reset_rtt_records(&self) {
        if let Some(ref rtt) = self.rtt {
            if let Ok(mut records) = rtt.rtt_records.lock() {
                *records = (0, 0.0);
            }
        }
    }

    pub fn statistics(&self) -> Option<Statistics> {
        let _now = SystemTime::now();
        if _now<self.start_timestamp || _now>self.stop_timestamp {
            return None;
        }

        let throughput = {
            match self.throttler.lock() {
                Err(_) => return None,
                Ok(throttler) => throttler.last_rate
            }
        };

        let rtt = {
            match &self.rtt {
                None => 0.0,
                Some(recorder) => match recorder.rtt_records.lock() {
                    Err(_) => return None,
                    Ok(val) => val.1/(val.0 as f64)
                }
            }
        };
        
        Some( Statistics{rtt,throughput} )
    }

    pub fn start(&mut self, index:usize) -> JoinHandle<()> {
        let throttler = Arc::clone(&self.throttler);
        let rtt_tx = match self.rtt {
            Some(ref mut rtt) => Some( rtt.start() ),
            None => None
        };
        let (StreamParam::UDP(ref params) | StreamParam::TCP(ref params)) = self.stream;
        let params = params.clone();
        let tx = self.tx.pop().unwrap();
        let blocked_signal = Arc::clone(&self.blocked_signal);

        let _now = SystemTime::now();
        self.start_timestamp = _now + Duration::from_secs_f64( params.duration[0] );
        self.stop_timestamp = _now + Duration::from_secs_f64( params.duration[1] );
        let source = thread::spawn(move || {
            source_thread(throttler, rtt_tx, params, tx, blocked_signal)
        });

        println!("{}. {} on ...", index, self.stream);
        source
    }
}
