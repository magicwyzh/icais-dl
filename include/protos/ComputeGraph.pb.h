// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ComputeGraph.proto

#ifndef PROTOBUF_INCLUDED_ComputeGraph_2eproto
#define PROTOBUF_INCLUDED_ComputeGraph_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3006001 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
#include "Tensor.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_ComputeGraph_2eproto

// Internal implementation detail -- do not use these members.
struct TableStruct_ComputeGraph_2eproto {
  static const ::google::protobuf::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors_ComputeGraph_2eproto();
namespace icdl_proto {
class GraphParams;
class GraphParamsDefaultTypeInternal;
extern GraphParamsDefaultTypeInternal _GraphParams_default_instance_;
class GraphParams_GraphParamsEntry_DoNotUse;
class GraphParams_GraphParamsEntry_DoNotUseDefaultTypeInternal;
extern GraphParams_GraphParamsEntry_DoNotUseDefaultTypeInternal _GraphParams_GraphParamsEntry_DoNotUse_default_instance_;
}  // namespace icdl_proto
namespace google {
namespace protobuf {
template<> ::icdl_proto::GraphParams* Arena::CreateMaybeMessage<::icdl_proto::GraphParams>(Arena*);
template<> ::icdl_proto::GraphParams_GraphParamsEntry_DoNotUse* Arena::CreateMaybeMessage<::icdl_proto::GraphParams_GraphParamsEntry_DoNotUse>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace icdl_proto {

// ===================================================================

class GraphParams_GraphParamsEntry_DoNotUse : public ::google::protobuf::internal::MapEntry<GraphParams_GraphParamsEntry_DoNotUse, 
    ::std::string, ::icdl_proto::Tensor,
    ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
    ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
    0 > {
public:
#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
static bool _ParseMap(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  typedef ::google::protobuf::internal::MapEntry<GraphParams_GraphParamsEntry_DoNotUse, 
    ::std::string, ::icdl_proto::Tensor,
    ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
    ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
    0 > SuperType;
  GraphParams_GraphParamsEntry_DoNotUse();
  GraphParams_GraphParamsEntry_DoNotUse(::google::protobuf::Arena* arena);
  void MergeFrom(const GraphParams_GraphParamsEntry_DoNotUse& other);
  static const GraphParams_GraphParamsEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const GraphParams_GraphParamsEntry_DoNotUse*>(&_GraphParams_GraphParamsEntry_DoNotUse_default_instance_); }
  void MergeFrom(const ::google::protobuf::Message& other) final;
  ::google::protobuf::Metadata GetMetadata() const;
};

// -------------------------------------------------------------------

class GraphParams : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:icdl_proto.GraphParams) */ {
 public:
  GraphParams();
  virtual ~GraphParams();

  GraphParams(const GraphParams& from);

  inline GraphParams& operator=(const GraphParams& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  GraphParams(GraphParams&& from) noexcept
    : GraphParams() {
    *this = ::std::move(from);
  }

  inline GraphParams& operator=(GraphParams&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor() {
    return default_instance().GetDescriptor();
  }
  static const GraphParams& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GraphParams* internal_default_instance() {
    return reinterpret_cast<const GraphParams*>(
               &_GraphParams_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(GraphParams* other);
  friend void swap(GraphParams& a, GraphParams& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline GraphParams* New() const final {
    return CreateMaybeMessage<GraphParams>(nullptr);
  }

  GraphParams* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<GraphParams>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const GraphParams& from);
  void MergeFrom(const GraphParams& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  static const char* _InternalParse(const char* begin, const char* end, void* object, ::google::protobuf::internal::ParseContext* ctx);
  ::google::protobuf::internal::ParseFunc _ParseFunc() const final { return _InternalParse; }
  #else
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(GraphParams* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  // map<string, .icdl_proto.Tensor> graph_params = 1;
  int graph_params_size() const;
  void clear_graph_params();
  static const int kGraphParamsFieldNumber = 1;
  const ::google::protobuf::Map< ::std::string, ::icdl_proto::Tensor >&
      graph_params() const;
  ::google::protobuf::Map< ::std::string, ::icdl_proto::Tensor >*
      mutable_graph_params();

  // @@protoc_insertion_point(class_scope:icdl_proto.GraphParams)
 private:
  class HasBitSetters;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::MapField<
      GraphParams_GraphParamsEntry_DoNotUse,
      ::std::string, ::icdl_proto::Tensor,
      ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
      ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
      0 > graph_params_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_ComputeGraph_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// -------------------------------------------------------------------

// GraphParams

// map<string, .icdl_proto.Tensor> graph_params = 1;
inline int GraphParams::graph_params_size() const {
  return graph_params_.size();
}
inline const ::google::protobuf::Map< ::std::string, ::icdl_proto::Tensor >&
GraphParams::graph_params() const {
  // @@protoc_insertion_point(field_map:icdl_proto.GraphParams.graph_params)
  return graph_params_.GetMap();
}
inline ::google::protobuf::Map< ::std::string, ::icdl_proto::Tensor >*
GraphParams::mutable_graph_params() {
  // @@protoc_insertion_point(field_mutable_map:icdl_proto.GraphParams.graph_params)
  return graph_params_.MutableMap();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace icdl_proto

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // PROTOBUF_INCLUDED_ComputeGraph_2eproto
