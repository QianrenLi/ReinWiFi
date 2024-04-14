use std::fs::File;
use std::io::prelude::*;
use std::time::SystemTime;
use std::collections::VecDeque;
use crate::packet::{PacketStruct,UDP_MAX_LENGTH};
// use std::sync::{Arc, Mutex};

type TIME = SystemTime;
type SIZE = usize;

static MAX_ERR_RATIO: f64 = 0.01;
static CYCLED_RATIO: usize = 50;

struct CycledVecDequeue<T> {
    size: usize,
    fifo: VecDeque<T>
}

impl<T> CycledVecDequeue<T>
where T:Sized + Copy
{
    pub fn new(size: usize) -> Self {
        let fifo = VecDeque::with_capacity(size);
        Self{ size, fifo }
    }

    pub fn push(&mut self, item: T) -> Option<T> {
        self.fifo.push_back(item);
        if self.size>0 && self.fifo.len()==self.size {
            self.fifo.pop_front()
        } else{ None }
    }

    pub fn try_push(&mut self, item: T) -> bool {
        if self.size>0 && self.fifo.len()==self.size {
            return false;
        }
        else {
            self.fifo.push_back(item);
            return true;
        }
    }

    pub fn len(&self) -> usize {
        self.fifo.len()
    }

    pub fn front(&self) -> Option<&T> {
        self.fifo.front()
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.fifo.pop_front()
    }

    pub fn reset(&mut self) {
        self.fifo.clear();
    }
}

pub struct RateThrottler {
    pub name: String,
    logger: Option<File>,
    window: CycledVecDequeue<(TIME, SIZE)>,
    buffer: CycledVecDequeue<PacketStruct>,
    sum_bytes: usize,
    acc_error: usize,
    max_error: usize,
    //
    pub throttle: f64,
    pub last_rate:f64,
}

impl RateThrottler {
    pub fn new(name:String, throttle: f64, window_size:usize, no_logging:bool, infinite_buffer:bool) -> Self {
        let buffer = match infinite_buffer {
            true  => CycledVecDequeue::new(0),
            false => CycledVecDequeue::new(CYCLED_RATIO * window_size)
        };
        let logger = match no_logging {
            false => Some(File::create( format!("logs/log-{}.txt", name) ).unwrap()),
            true => None
        };
        let window = CycledVecDequeue::new(window_size);
        let max_error = (MAX_ERR_RATIO * window_size as f64) as usize * UDP_MAX_LENGTH;

        // let last_rate = Arc::new(Mutex::new( 0.0 ));
        // let throttle = Arc::new(Mutex::new( throttle ));

        Self{ name, logger, window, buffer, throttle, last_rate:0.0,
                sum_bytes:0, acc_error:0, max_error }
    }

    pub fn reset(&mut self) {
        self.last_rate = 0.0;
        (self.sum_bytes,self.acc_error,self.max_error) = (0,0,0);
        self.window.reset();
        self.buffer.reset();
    }

    pub fn current_rate_mbps(&mut self, extra_bytes:Option<usize>) -> Option<f64> {
        if self.acc_error < self.max_error {
            return Some(self.last_rate);
            // if let Ok(rate)=self.last_rate.try_lock() { return Some(rate.clone()); }
        } else {
            self.acc_error = 0;
        }

        let acc_size = self.sum_bytes + extra_bytes.unwrap_or(0);

        let _last_time = self.window.front()?.0;
        let acc_time = SystemTime::now().duration_since( _last_time ).unwrap();
        let acc_time = acc_time.as_nanos();

        let average_rate_mbps = 8.0 * (acc_size as f64/1e6) / (acc_time as f64*1e-9);
        self.last_rate = average_rate_mbps;
        // if let Ok(mut ref_rate) = self.last_rate.try_lock() {
        //     *ref_rate = average_rate_mbps;
        // }
        Some(average_rate_mbps)
    }

    pub fn prepare(&mut self, packets: Vec<PacketStruct>) {
        let timestamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f64();
        let _rate_mbps = self.current_rate_mbps(None).unwrap_or(0.0);
        if let Some(ref mut logger) = self.logger {
            let message = format!("{:.9} {} {:.6}\n", timestamp, self.buffer.len(), _rate_mbps );
            logger.write_all( message.as_bytes() ).unwrap();
        }
        for packet in packets.into_iter() {
            self.buffer.try_push(packet);
        }
    }

    pub fn try_consume<T>(&mut self, callback:T) -> Option<bool>
    where T: Fn(PacketStruct) -> bool {
        match self.buffer.front().cloned() {
            None => None,
            Some(packet) => {
                if self.exceeds_with(packet.length as usize) {
                    std::thread::sleep( std::time::Duration::from_nanos(100_000) );
                    return Some(false);
                }
                match callback(packet) {
                    true => {
                        self.consume();
                        Some(true)
                    }
                    false => Some(false)
                }
            }
        }
    }

    pub fn consume(&mut self) -> Option<PacketStruct> {
        let timestamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f64();
        let _rate_mbps = self.current_rate_mbps(None).unwrap_or(0.0);
        if let Some(ref mut logger) = self.logger {
            let message = format!("{:.9} {} {:.6}\n", timestamp, self.buffer.len(), _rate_mbps );
            logger.write_all( message.as_bytes() ).unwrap();
        }
        self.buffer.pop_front()
    }

    pub fn exceeds_with(&mut self, size_bytes:usize) -> bool {
        let _throttle = self.throttle;//.lock().unwrap().clone();

        if _throttle==0.0 || self.window.len()==0 {
            self.sum_bytes += size_bytes;
            if let Some(item) = self.window.push(( SystemTime::now(), size_bytes )) {
                self.sum_bytes -= item.1 as usize;
                self.acc_error += item.1;
            }
            return false;
        }

        self.acc_error += size_bytes;

        let average_rate_mbps = self.current_rate_mbps( Some(size_bytes) );
        if average_rate_mbps.unwrap() < _throttle {
            self.sum_bytes += size_bytes;
            if let Some(item) = self.window.push(( SystemTime::now(), size_bytes )) {
                self.sum_bytes -= item.1 as usize;
                self.acc_error += item.1;
            }
            false
        }
        else {
            true
        }
    }
}
