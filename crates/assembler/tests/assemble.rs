use tritium_assembler::assemble;

#[test]
fn load_immediate_expands_when_needed() {
    assert_eq!(assemble("li r1, -5").unwrap().len(), 1);
    assert_eq!(assemble("li r1, 0x12345678").unwrap().len(), 2);
    assert_eq!(assemble("mov r1, r2").unwrap(), assemble("add r1, r2, r0").unwrap());
}
