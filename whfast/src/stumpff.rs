const FACTORIALS: [f64; 24] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
    51090942171709440000.0,
    1124000727777607680000.0,
    25852016738884976640000.0,
];

pub fn stumpff(mut z: f64, n: u8) -> f64 {
    let mut n_counter = 0;
    while z > 0.1 {
        z /= 4.0;
        n_counter += 1;
    }
    let mut c_4 = 1.0 / FACTORIALS[4] - z * 1.0 / FACTORIALS[6];

    let mut c_5 = 1.0 / FACTORIALS[5] - z * 1.0 / FACTORIALS[7];

    let z_bar = -z;

    let mut p = z_bar;
    let mut k = 8;
    loop {
        let c_4_prev = c_4;
        p = p * z_bar;
        c_4 = c_4 + p / FACTORIALS[k];
        k += 1;
        c_5 = c_5 + p / FACTORIALS[k];
        k += 1;
        if c_4 == c_4_prev {
            break;
        }
    }
    let mut c_3 = 1.0 / 6.0 - z * c_5;

    let mut c_2 = 1.0 / 2.0 - z * c_4;
    let mut c_1 = 1.0 - z * c_3;

    while n_counter > 0 {
        z = 4.0 * z;
        c_5 = 1.0 / 16.0 * (c_5 + c_4 + c_3 + c_2);
        c_4 = 1.0 / 8.0 * c_3 * (1.0 - c_1);
        c_3 = 1.0 / 6.0 - z * c_5;
        c_2 = 1.0 / 2.0 - z * c_3;
        c_1 = 1.0 - z * c_3;
        n_counter -= 1;
    }
    let c_0 = 1.0 - z * c_2;
    match n {
        0 => c_0,
        1 => c_1,
        2 => c_2,
        3 => c_3,
        4 => c_4,
        5 => c_5,
        _ => panic!("Invalid Stumpff index!"),
    }
}
