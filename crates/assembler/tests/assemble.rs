use tritium_assembler::{assemble, disassemble};
use tritium_isa::{
    BranchCondition, ImmediateOperation, LoadWidth, RegisterOperation, Special, StoreWidth,
};

fn sample_program() -> String {
    let mut lines = vec!["top:".to_string()];
    lines.extend(
        RegisterOperation::ALL
            .iter()
            .map(|operation| format!("{} r1, r2, r3", operation.mnemonic())),
    );
    lines.extend(
        ImmediateOperation::ALL
            .iter()
            .map(|operation| format!("{} r4, r5, 7", operation.mnemonic())),
    );
    lines.push("addi r4, r5, -7".to_string());
    for suffix in ["", "s"] {
        lines.extend(
            LoadWidth::ALL.iter().map(|width| format!("{}{suffix} r6, -4(r7)", width.mnemonic())),
        );
        lines.extend(
            StoreWidth::ALL.iter().map(|width| format!("{}{suffix} r8, 12(r9)", width.mnemonic())),
        );
    }
    lines.extend(
        BranchCondition::ALL
            .iter()
            .map(|condition| format!("{} r10, r11, top", condition.mnemonic())),
    );
    lines.extend(Special::ALL.iter().map(|special| format!("sreg r12, {}", special.mnemonic())));
    lines.extend([
        "lui r13, 0xBEEF".to_string(),
        "jal r31, top".to_string(),
        "jalr r0, r31, 0".to_string(),
        "bar".to_string(),
        "exit".to_string(),
    ]);
    lines.join("\n")
}

#[test]
fn labels_resolve_forward_and_backward() {
    let words =
        assemble("start:\naddi r1, r1, 1\nbne r1, r2, start\nj end\nadd r3, r3, r3\nend:\nexit")
            .unwrap();
    let listing = disassemble(&words).unwrap();
    assert_eq!(listing.lines().nth(1), Some("bne r1, r2, -1"));
    assert_eq!(listing.lines().nth(2), Some("jal r0, 2"));
}

#[test]
fn load_immediate_expands_when_needed() {
    assert_eq!(assemble("li r1, -5").unwrap().len(), 1);
    assert_eq!(assemble("li r1, 0x12345678").unwrap().len(), 2);
    assert_eq!(assemble("mov r1, r2").unwrap(), assemble("add r1, r2, r0").unwrap());
}

#[test]
fn errors_name_offending_line() {
    assert_eq!(assemble("add r1, r2, r3\nfrobnicate r1").unwrap_err().line, 2);
    assert_eq!(assemble("addi r1, r2, 40000").unwrap_err().line, 1);
    assert_eq!(assemble("add r1, r2, r32").unwrap_err().line, 1);
    assert_eq!(assemble("beq r1, r2, nowhere").unwrap_err().line, 1);
    assert_eq!(assemble("add r1, r2").unwrap_err().line, 1);
    assert_eq!(assemble("loop:\nloop:\nexit").unwrap_err().line, 2);
}

#[test]
fn far_branch_targets_are_rejected() {
    let source = format!("far:\n{}beq r1, r2, far", "add r1, r1, r1\n".repeat(40000));
    assert_eq!(assemble(&source).unwrap_err().line, 40002);
}

#[test]
fn disassembly_reassembles_same_words() {
    let words = assemble(&sample_program()).unwrap();
    assert_eq!(assemble(&disassemble(&words).unwrap()).unwrap(), words);
}

#[test]
fn error_messages_include_line_number() {
    let error = assemble("exit\nadd r1").unwrap_err();
    assert!(error.to_string().starts_with("line 2: "), "{error}");
}

#[test]
fn load_immediate_keeps_unsigned_halfwords() {
    let words = assemble("li r1, 40000").unwrap();
    assert_eq!(words, assemble("lui r1, 0\nori r1, r1, 0x9C40").unwrap());
}
