use std::thread::{JoinHandle, sleep, yield_now};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::packet::{PacketStruct, PacketSender, PacketReceiver, tos2ac};
use crate::dispatcher::{UdpDispatcher, SourceInput};

type BrokerConn = (PacketReceiver, PacketSender);
struct Application {
    conn: BrokerConn,
    priority: String
}
type GuardedApplications = Arc<Mutex<Vec<Application>>>;

fn policy_priority_fifo(all_apps: Vec<GuardedApplications>) {
    let passthrough = |app:&Application| {
        let _ = app.priority;
        for packet in app.conn.0.try_iter() {
            if let Err(_) = app.conn.1.send(packet) {
                return false;
            }
        }
        if app.priority.contains("guarded,") {
            sleep( Duration::from_micros(10) );
        }
        true
    };

    loop {
        for ac_apps in all_apps.iter() {
            yield_now();
            ac_apps.lock().unwrap().retain(passthrough);
            sleep( Duration::from_micros(10) );
        }
    }
}

pub struct GlobalBroker {
    name: Option<String>,
    ipaddr: String,
    use_agg_socket: Option<bool>,
    pub dispatcher: UdpDispatcher,
    apps: [GuardedApplications; 4]
}

impl GlobalBroker {
    pub fn new(name:Option<String>, ipaddr:String, use_agg_socket:Option<bool>) -> Self {
        let apps = [
            Arc::new(Mutex::new( Vec::<Application>::new() )),
            Arc::new(Mutex::new( Vec::<Application>::new() )),
            Arc::new(Mutex::new( Vec::<Application>::new() )),
            Arc::new(Mutex::new( Vec::<Application>::new() ))
        ];
        let dispatcher = UdpDispatcher::new();

        Self { name, ipaddr, use_agg_socket, dispatcher, apps }
    }

    pub fn add(&mut self, tos: u8, priority: String) -> SourceInput {
        let ac = tos2ac( tos );

        let (broker_tx, blocked_signal) = match self.use_agg_socket {
            Some(false) | None => self.dispatcher.start_new(self.ipaddr.clone(), tos),
            Some(true) => {
                let (tx, blocked_signal) = self.dispatcher.records.get(ac).unwrap();
                ( tx.clone(), Arc::clone(&blocked_signal) )
            }
        };

        match self.name {
            None => (broker_tx, blocked_signal),
            Some(_) => {
                let (tx, broker_rx) = mpsc::channel::<PacketStruct>();

                let conn = (broker_rx, broker_tx);
                let app = Application{ conn, priority };
                self.apps[ac].lock().unwrap().push( app );

                (tx, blocked_signal)
            }
        }
    }

    pub fn start(&mut self) -> JoinHandle<()> {
        let apps:Vec<_> = self.apps.iter().map(|app| app.clone()).collect();

        if let Some(true) = self.use_agg_socket {
            self.dispatcher.start_agg_sockets( String::new() );
        }

        std::thread::spawn(move || {
            policy_priority_fifo(apps);
        })
    }

}
