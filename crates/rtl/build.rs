use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) -> String {
    let output = command.output().unwrap_or_else(|error| panic!("{command:?}: {error}"));
    if !output.status.success() {
        panic!(
            "{command:?} failed\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn sources(directory: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "sv"))
        .collect();
    paths.sort_by_key(|path| (!path.ends_with("tritium_package.sv"), path.clone()));
    paths
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let hardware = manifest.join("../../hardware");
    let rtl = hardware.join("rtl");
    let harness = hardware.join("verilator/harness.cpp");
    let output = PathBuf::from(env::var("OUT_DIR").unwrap());
    let model = output.join("model");
    let deployment_target =
        env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_string());
    println!("cargo:rerun-if-changed={}", hardware.display());
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    run(Command::new("verilator")
        .env("MACOSX_DEPLOYMENT_TARGET", &deployment_target)
        .args(["--cc", "--build", "-j", "0", "-O3", "--top-module", "tritium_gpu"])
        .args(["-Wall", "-Wno-UNUSEDPARAM", "-Wno-UNUSEDSIGNAL", "-CFLAGS", "-fPIC"])
        .arg("--Mdir")
        .arg(&model)
        .arg(format!("-I{}", rtl.display()))
        .args(sources(&rtl)));

    let root = PathBuf::from(run(Command::new("verilator").args(["--getenv", "VERILATOR_ROOT"])));
    let object = output.join("harness.o");
    run(Command::new("c++")
        .env("MACOSX_DEPLOYMENT_TARGET", &deployment_target)
        .args(["-std=c++17", "-O2", "-fPIC", "-c"])
        .arg(format!("-I{}", model.display()))
        .arg(format!("-I{}", root.join("include").display()))
        .arg(format!("-I{}", root.join("include/vltstd").display()))
        .arg(&harness)
        .arg("-o")
        .arg(&object));
    run(Command::new("ar").arg("rcs").arg(output.join("libtritium_harness.a")).arg(&object));
    fs::copy(model.join("Vtritium_gpu__ALL.a"), output.join("libtritium_model.a")).unwrap();
    fs::copy(model.join("libverilated.a"), output.join("libtritium_verilated.a")).unwrap();

    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=tritium_harness");
    println!("cargo:rustc-link-lib=static=tritium_model");
    println!("cargo:rustc-link-lib=static=tritium_verilated");
    let standard_library =
        if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" { "c++" } else { "stdc++" };
    println!("cargo:rustc-link-lib=dylib={standard_library}");
}
