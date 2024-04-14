mod socket;

use pyo3::prelude::*;
use std::borrow::Cow;
use std::net::UdpSocket;
use crate::socket::*;

#[pyclass]
struct PriorityTxSocket {
    sock: UdpSocket
}

#[pymethods]
impl PriorityTxSocket {
    #[new]
    fn new(tos:u8) -> Self {
        let sock = create_udp_socket(tos, None).unwrap();
        Self { sock }
    }

    #[pyo3(name = "sendto")]
    fn send_to(&self, buf:Cow<[u8]>, addr_port:(String,u16)) {
        let addr = format!("{}:{}", addr_port.0, addr_port.1);
        self.sock.send_to(&buf, &addr).unwrap();
    }
}

#[pymodule]
fn replay(_py:Python, m:&PyModule) -> PyResult<()> {
    m.add_class::<PriorityTxSocket>()?;
    Ok(())
}
