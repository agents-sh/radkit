use std::env;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = Path::new("proto");
    let descriptor_path = PathBuf::from(env::var("OUT_DIR")?).join("a2a-file-descriptor-set.bin");

    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    config.file_descriptor_set_path(&descriptor_path);
    config.extern_path(".google.protobuf.Struct", "::pbjson_types::Struct");
    config.extern_path(".google.protobuf.Value", "::pbjson_types::Value");
    config.extern_path(".google.protobuf.ListValue", "::pbjson_types::ListValue");
    config.extern_path(".google.protobuf.Timestamp", "::pbjson_types::Timestamp");

    config.compile_protos(&[proto_root.join("a2a.proto")], &[proto_root])?;

    let descriptor_set = std::fs::read(descriptor_path)?;
    pbjson_build::Builder::new()
        .register_descriptors(&descriptor_set)?
        .build(&[".lf.a2a.v1"])?;

    println!("cargo:rerun-if-changed=proto/a2a.proto");
    println!("cargo:rerun-if-changed=proto/google/api/annotations.proto");
    println!("cargo:rerun-if-changed=proto/google/api/client.proto");
    println!("cargo:rerun-if-changed=proto/google/api/field_behavior.proto");
    println!("cargo:rerun-if-changed=proto/google/api/http.proto");

    Ok(())
}
