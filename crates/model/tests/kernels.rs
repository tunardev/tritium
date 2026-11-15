#[test]
fn all_model_kernels_compile() {
    let kernels = tritium_compiler::compile(tritium_model::KERNELS).unwrap();
    let names: Vec<&str> = kernels.iter().map(|kernel| kernel.name.as_str()).collect();
    assert_eq!(names, ["embed", "quantize", "project", "attention", "head"]);
}
