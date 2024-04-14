mod conf;
mod packet;
mod throttle;
mod broker;
mod source;
mod dispatcher;
mod rtt;
mod socket;
mod ipc;

use std::collections::HashMap;
use std::path::Path;
use clap::Parser;
use serde_json;

use crate::conf::Manifest;
use crate::ipc::IPCDaemon;
use crate::source::SourceManager;
use crate::broker::GlobalBroker;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct ProgArgs {
    /// The manifest file tied with the data trace.
    #[clap( value_parser )]
    manifest_file: String,
    /// The target server IP address.
    #[clap( value_parser )]
    target_ip_address: String,
    /// The duration of test procedure (unit: seconds).
    #[clap( value_parser )]
    duration: f64,
    /// IPC Port for real-time access
    #[clap(long, default_value_t = 11112)]
    ipc_port: u16,
}

fn main() {
    // load the manifest file
    let args = ProgArgs::parse();
    let ipaddr = args.target_ip_address;
    let file = std::fs::File::open(&args.manifest_file).unwrap();
    let reader = std::io::BufReader::new( file );
    let root = Path::new(&args.manifest_file).parent();
    let manifest:Manifest = serde_json::from_reader(reader).unwrap();
    // parse the manifest file
    let streams:Vec<_> = manifest.streams.into_iter().filter_map( |x| x.validate(root, args.duration) ).collect();
    let window_size = manifest.window_size;
    let orchestrator = manifest.orchestrator;
    println!("Sliding Window Size: {}.", window_size);
    println!("Orchestrator: {:?}.", orchestrator);

    // start broker
    let mut broker = GlobalBroker::new( orchestrator, ipaddr, manifest.use_agg_socket );
    let _handle = broker.start();

    // spawn the source thread
    let mut sources:HashMap<_,_> = streams.into_iter().map(|stream| {
        let src = SourceManager::new(stream, window_size, &mut broker);
        let name = src.name.clone();
        (name, src)
    }).collect();
    let _handles:Vec<_> = sources.iter_mut().enumerate().map(|(i,(_name,src))| {
        src.start(i+1)
    }).collect();

    // start global IPC
    let ipc = IPCDaemon::new( sources, args.ipc_port );
    ipc.start_loop( args.duration );

    std::process::exit(0); //force exit
}
