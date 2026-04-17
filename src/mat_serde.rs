//! Custom serde support for `faer::Mat<f64>`.

use faer::Mat;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
struct MatProxy {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

pub fn serialize<S: Serializer>(mat: &Mat<f64>, s: S) -> Result<S::Ok, S::Error> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            data.push(mat[(i, j)]);
        }
    }
    MatProxy { nrows, ncols, data }.serialize(s)
}

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Mat<f64>, D::Error> {
    use serde::de::Error;
    let proxy = MatProxy::deserialize(d)?;
    let expected = proxy.nrows * proxy.ncols;
    if proxy.data.len() != expected {
        return Err(D::Error::invalid_length(
            proxy.data.len(),
            &format!("{expected}").as_str(),
        ));
    }
    let mut mat = Mat::<f64>::zeros(proxy.nrows, proxy.ncols);
    for i in 0..proxy.nrows {
        for j in 0..proxy.ncols {
            mat[(i, j)] = proxy.data[i * proxy.ncols + j];
        }
    }
    Ok(mat)
}
