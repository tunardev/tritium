mod support;

use support::{SMALL, random_model};
use tritium_model::{ModelConfig, ReferenceSession};

#[test]
fn reference_logits_vary_with_context() {
    let model = random_model(SMALL, 7);
    let mut first = ReferenceSession::new(&model);
    let mut second = ReferenceSession::new(&model);
    first.decode(1).unwrap();
    second.decode(2).unwrap();
    assert_ne!(first.decode(5).unwrap(), second.decode(5).unwrap());
}

#[test]
fn full_context_is_an_error() {
    let model = random_model(SMALL, 7);
    let mut session = ReferenceSession::new(&model);
    for _ in 0..SMALL.context {
        session.decode(0).unwrap();
    }
    assert!(session.decode(0).is_err());
}

#[test]
fn invalid_shapes_are_rejected() {
    let config = ModelConfig { width: 96, ..SMALL };
    assert!(config.validate().is_err());
    assert!(ModelConfig { heads: 3, ..SMALL }.validate().is_err());
    assert!(SMALL.validate().is_ok());
}
