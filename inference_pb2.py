# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: inference.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'inference.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12\tinference\"$\n\x15InstanceDetectorInput\x12\x0b\n\x03url\x18\x01 \x01(\t\")\n\x16InstanceDetectorOutput\x12\x0f\n\x07objects\x18\x01 \x03(\t2b\n\x10InstanceDetector\x12N\n\x07Predict\x12 .inference.InstanceDetectorInput\x1a!.inference.InstanceDetectorOutputb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_INSTANCEDETECTORINPUT']._serialized_start=30
  _globals['_INSTANCEDETECTORINPUT']._serialized_end=66
  _globals['_INSTANCEDETECTOROUTPUT']._serialized_start=68
  _globals['_INSTANCEDETECTOROUTPUT']._serialized_end=109
  _globals['_INSTANCEDETECTOR']._serialized_start=111
  _globals['_INSTANCEDETECTOR']._serialized_end=209
# @@protoc_insertion_point(module_scope)
