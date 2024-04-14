use std::{collections::HashMap, time::{Duration, SystemTime}};
use serde::{Serialize, Deserialize};
use crate::source::SourceManager;
use crate::socket::create_udp_socket;

#[derive(Serialize, Deserialize, Debug,Clone)]
pub struct Statistics {
    pub rtt: f64,
    pub throughput: f64,
}

#[derive(Serialize, Deserialize, Debug,Clone)]
struct Request {
    cmd: String,
    body: Option<HashMap<String,f64>>
}

#[derive(Serialize, Deserialize, Debug,Clone)]
struct Response {
    cmd: String,
    body: Option<HashMap<String,Statistics>>
}

pub struct IPCDaemon {
    ipc_port: u16,
    sources: HashMap<String, SourceManager>
}

impl IPCDaemon {
    pub fn new(sources: HashMap<String, SourceManager>, ipc_port: u16) -> Self {
        Self{ sources, ipc_port }
    }

    fn handle_request(&self, req:Request) -> Option<Response> {
        let cmd = req.cmd.clone();

        match req.cmd.as_str() {
            "throttle" => {
                if let Some(data) = req.body {
                    let _:Vec<_> = data.iter().map(|(name, value)| {
                        self.sources[name].throttle(*value);
                    }).collect();
                }
                // reset RTT records for all
                let _:Vec<_> = self.sources.iter().map(|(_,src)| {
                    src.reset_rtt_records()
                }).collect();
                //
                return None;
            },

            "statistics" => {
                let body = Some( self.sources.iter().filter_map(|(name,src)| {
                    match src.statistics() {
                        Some(stat) => Some(( name.clone(), stat )),
                        None => None
                    }
                }).collect() );
                //
                return Some(Response{ cmd, body });
            },

            _ => { return None; }
        }
    }

    pub fn start_loop(&self, duration:f64) {
        let deadline = SystemTime::now() + Duration::from_secs_f64(duration);
        let addr = format!("0.0.0.0:{}", self.ipc_port);
        let sock = create_udp_socket(192, Some(addr)).unwrap();
        // let sock = UdpSocket::bind(&addr).unwrap();
        sock.set_nonblocking(true).unwrap();
        let mut buf = [0; 2048];

        while SystemTime::now() < deadline {
            if let Ok((len, src_addr)) = sock.recv_from(&mut buf) {
                let buf_str = std::str::from_utf8(&buf[..len]).unwrap();
                let req = serde_json::from_str::<Request>(buf_str).unwrap();
                if let Some(res) = self.handle_request(req) {
                    let res = serde_json::to_string(&res).unwrap();
                    sock.send_to(res.as_bytes(), src_addr).unwrap();
                }
            }
            std::thread::sleep( Duration::from_nanos(10_000_000) );
        }
    }
}
