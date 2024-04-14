use std::fs::File;
use std::io::prelude::*;
use std::collections::HashMap;
use std::net::UdpSocket;
use std::thread::{self, JoinHandle};
use std::sync::{Arc, Mutex, mpsc};
use std::time::SystemTime;

type SeqRecords = HashMap<u32,f64>;
type GuardedSeqRecords = Arc<Mutex<SeqRecords>>;
type RttRecords = (usize, f64);
type GuardedRttRecords = Arc<Mutex<RttRecords>>;
pub type RttSender = mpsc::Sender<u32>;
type RttReceiver = mpsc::Receiver<u32>;
static PONG_PORT_INC:u16 = 1024;

pub struct RttRecorder {
    record_handle: Option<JoinHandle<()>>,
    recv_handle: Option<JoinHandle<()>>,
    name: String,
    port: u16,
    pub rtt_records: GuardedRttRecords
}

fn update_rtt_records(records:&GuardedRttRecords, rtt:f64) {
    if let Ok(mut records) = records.lock() {
        let (len, sum_rtt) = *records;
        *records = (len+1, sum_rtt+rtt);
    }
}

fn record_thread(rx: RttReceiver, records: GuardedSeqRecords) {
    while let Ok(seq) = rx.recv() {
        let time_now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f64();
        let mut _records = records.lock().unwrap();
        _records.insert(seq, time_now);
    }
}

fn pong_recv_thread(name: String, port: u16, seq_records: GuardedSeqRecords, rtt_records: GuardedRttRecords) {
    let mut buf = [0; 2048];
    let sock = UdpSocket::bind( format!("0.0.0.0:{}", port)).unwrap();
    let mut logger = File::create( format!("logs/rtt-{}.txt", name) ).unwrap();

    while let Ok(_) = sock.recv_from(&mut buf) {
        let msg: [u8;4] = buf[..4].try_into().unwrap();
        let seq = u32::from_le_bytes( msg );
        let _msg: [u8;8] = buf[10..18].try_into().unwrap();
        let _duration = f64::from_le_bytes( _msg );
        let time_now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f64();
        if let Some(last_time) = {
            let mut _records = seq_records.lock().unwrap();
            _records.remove(&seq)
        } {
            let rtt = time_now - last_time;
            update_rtt_records(&rtt_records, rtt);
            let message = format!("{} {:.6}\n", seq, rtt);
            logger.write_all( message.as_bytes() ).unwrap();
        };
    }
}

impl RttRecorder {
    pub fn new(name:&String, port:u16) -> Self {
        let name = name.clone();
        let port = port + PONG_PORT_INC; //pong recv port
        let record_handle = None;
        let recv_handle = None;
        let rtt_records = Arc::new(Mutex::new( (0,0.0) ));

        RttRecorder{ name, port, record_handle, recv_handle, rtt_records }
    }

    pub fn start(&mut self) -> RttSender {
        let (tx, rx) = mpsc::channel::<u32>();
        let (name, port) = (self.name.clone(), self.port);
        let seq_records1: GuardedSeqRecords = Arc::new(Mutex::new(HashMap::new()));
        let seq_records2 = seq_records1.clone();
        let rtt_records  = Arc::clone(&self.rtt_records);
        
        self.record_handle = Some(
            thread::spawn(move || { record_thread(rx, seq_records1); })
        );
        self.recv_handle = Some(
            thread::spawn(move || { pong_recv_thread(name, port, seq_records2, rtt_records); } )
        );

        tx
    }

    
}
