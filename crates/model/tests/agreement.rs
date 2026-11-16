mod support;

use support::{SMALL, random_model};
use tritium_model::{DeviceSession, ModelConfig, ReferenceSession, WeightFormat};
use tritium_runtime::BackendKind;

fn assert_backend_matches_reference(
    config: ModelConfig,
    kind: BackendKind,
    format: WeightFormat,
    tokens: &[usize],
) {
    let model = random_model(config, 7);
    let mut reference = ReferenceSession::new(model.clone());
    let mut device = DeviceSession::new(&model, kind, format);
    for (position, &token) in tokens.iter().enumerate() {
        let expected = reference.decode(token).unwrap();
        let actual = device.decode(token).unwrap();
        assert_eq!(actual, expected, "{kind} differs at position {position}");
    }
}

#[test]
fn simulator_logits_match_reference() {
    let tokens = [3, 17, 63, 0, 42, 42, 9, 1];
    assert_backend_matches_reference(SMALL, BackendKind::Simulator, WeightFormat::Ternary, &tokens);
}

#[test]
fn rtl_logits_match_reference() {
    assert_backend_matches_reference(SMALL, BackendKind::Rtl, WeightFormat::Ternary, &[5, 60, 12]);
}

#[test]
fn int8_weights_give_identical_logits() {
    let tokens = [8, 1, 33, 33];
    assert_backend_matches_reference(SMALL, BackendKind::Simulator, WeightFormat::Int8, &tokens);
    assert_backend_matches_reference(SMALL, BackendKind::Rtl, WeightFormat::Int8, &tokens[..2]);
}

#[test]
fn int8_weights_quadruple_the_bytes() {
    let model = random_model(SMALL, 7);
    let ternary = model.projection_weight_bytes(WeightFormat::Ternary);
    let int8 = model.projection_weight_bytes(WeightFormat::Int8);
    assert!(int8 >= ternary * 4 && int8 <= ternary * 5, "{ternary} vs {int8}");
}

#[test]
fn reference_logits_vary_with_context() {
    let model = random_model(SMALL, 7);
    let mut first = ReferenceSession::new(model.clone());
    let mut second = ReferenceSession::new(model.clone());
    first.decode(1).unwrap();
    second.decode(2).unwrap();
    assert_ne!(first.decode(5).unwrap(), second.decode(5).unwrap());
}

#[test]
fn full_context_is_an_error() {
    let model = random_model(SMALL, 7);
    let mut session = ReferenceSession::new(model.clone());
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
