use std::path::Path;
use rand::prelude::*;
use rand::distributions::Standard;
use serde::{Serialize, Deserialize};

const fn _default_duration() -> [f64; 2] { [0.0, f64::MAX] }
const fn _default_loops() -> usize { usize::MAX }
fn _random_value<T>() -> T where Standard: Distribution<T> { rand::thread_rng().gen() }
#[derive(Serialize, Deserialize, Debug,Clone)]
pub struct ConnParams {
    pub npy_file: String,
    #[serde(default = "_random_value")]     //default:
    pub port: u16,                          //         <random>
    #[serde(default = "_default_duration")] //default:
    pub duration: [f64; 2],                 //         [0.0, +inf]
    #[serde(default = "_random_value")]     //default:
    pub start_offset: usize,                //         <random>
    #[serde(default = "_default_loops")]    //default:
    pub loops: usize,                       //         +inf
    #[serde(default)] pub tos: u8,          //default: 0
    #[serde(default)] pub throttle: f64,    //default: 0.0
    #[serde(default)] pub priority: String, //default: ""
    #[serde(default)] pub calc_rtt: bool,   //default: false
    #[serde(default)] pub no_logging: bool, //default: false
}


#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum StreamParam {
    TCP(ConnParams),
    UDP(ConnParams)
}

impl std::fmt::Display for StreamParam {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (_type, _param) =
        match self {
            Self::TCP(p) => ("TCP", p),
            Self::UDP(p) => ("UDP", p)
        };
        let _file:String = _param.npy_file.clone();

        write!(f,
            "{type} {{ port: {port}, tos: {tos}, throttle: {throttle} Mbps, file: \"{file}\", loops: {loops} }}",
            type=_type, port=_param.port, tos=_param.tos, throttle=_param.throttle, loops=_param.loops as isize, file=_file
        )
    }
}

impl StreamParam {
    pub fn validate(mut self, root:Option<&Path>, duration:f64) -> Option<Self> {
        let ( Self::TCP(ref mut param) | Self::UDP(ref mut param) ) = self;

        // validate npy file existence
        let cwd = std::env::current_dir().unwrap();
        let path_trail1 = cwd.join( &param.npy_file );
        let path_trail2 = root.unwrap_or( cwd.as_path() ).join( &param.npy_file );
        if path_trail1.exists() {
            param.npy_file = String::from( path_trail1.to_str().unwrap() );
        }
        else if path_trail2.exists() {
            param.npy_file = String::from( path_trail2.to_str().unwrap() );
        }
        else {
            return None;
        }

        // validate duration
        if param.duration[1] > duration {
            param.duration[1] = duration;
        }

        Some(self)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Manifest {
    pub use_agg_socket: Option<bool>,
    pub orchestrator: Option<String>,
    pub window_size: usize,
    pub streams: Vec<StreamParam>
}
