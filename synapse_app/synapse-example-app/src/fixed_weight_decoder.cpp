#include "fixed_weight_decoder.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <synapse-app-sdk/middleware/conversions.hpp>

namespace app {

// ============================================================
// Training-derived constants (from analysis/decoder_v7/feat_norm.npz)
// ============================================================

// Channel standard deviations from training (64 channels)
// Used to compute spike thresholds: threshold = sigma * channel_std
static constexpr float kChannelStds[64] = {
  17.694620f, 20.075947f, 19.335461f, 22.877888f, 17.090914f, 17.635862f, 23.494164f, 19.091438f,
  20.701656f, 22.650154f, 21.520803f, 19.253656f, 23.650700f, 22.415810f, 16.899929f, 20.350985f,
  19.953016f, 23.618177f, 22.962992f, 18.056860f, 20.367434f, 23.385099f, 24.251097f, 23.360914f,
  18.681084f, 20.261679f, 17.712212f, 21.622915f, 22.037830f, 20.483042f, 18.776751f, 18.094400f,
  21.777956f, 23.510193f, 21.308781f, 17.024895f, 18.335258f, 22.690905f, 23.349989f, 22.312614f,
  21.717251f, 21.178856f, 17.268847f, 22.569601f, 23.706341f, 17.139065f, 21.096640f, 22.738298f,
  16.620424f, 20.400780f, 19.963533f, 21.702463f, 19.939156f, 22.393656f, 23.928518f, 22.519955f,
  19.029495f, 21.041216f, 21.254198f, 22.699968f, 19.719887f, 19.429047f, 23.914284f, 18.162258f,
};

// Feature means from training (192 features = 64 channels x 3 thresholds)
static constexpr float kFeatMean[192] = {
  1.362340f, 2.234871f, 2.013060f, 3.092341f, 1.153850f, 1.384878f, 3.311804f, 1.895712f,
  2.484645f, 3.024259f, 2.682266f, 1.979654f, 3.368566f, 2.964855f, 1.112340f, 2.329142f,
  2.204490f, 3.353906f, 3.127939f, 1.508173f, 2.332551f, 3.270677f, 3.499930f, 3.270816f,
  1.795910f, 2.306813f, 1.398372f, 2.736905f, 2.882808f, 2.383069f, 1.809092f, 1.535128f,
  2.783267f, 3.326795f, 2.619574f, 1.135086f, 1.642842f, 3.063561f, 3.274659f, 2.960490f,
  2.791562f, 2.585716f, 1.233618f, 3.031841f, 3.381921f, 1.183396f, 2.569091f, 3.058170f,
  0.999113f, 2.367505f, 2.214350f, 2.757582f, 2.240418f, 3.011964f, 3.423970f, 3.001061f,
  1.862966f, 2.565109f, 2.631313f, 3.043267f, 2.127000f, 2.056883f, 3.427935f, 1.568830f,
  0.913693f, 1.595663f, 1.405294f, 2.422005f, 0.755356f, 0.928996f, 2.650633f, 1.313613f,
  1.798153f, 2.347280f, 2.002313f, 1.373417f, 2.715289f, 2.283250f, 0.718420f, 1.671118f,
  1.560796f, 2.694978f, 2.456403f, 1.021564f, 1.665623f, 2.608862f, 2.880408f, 2.609314f,
  1.235079f, 1.645277f, 0.939569f, 2.039545f, 2.186613f, 1.709342f, 1.245183f, 1.041754f,
  2.095924f, 2.664632f, 1.936909f, 0.741792f, 1.123504f, 2.378965f, 2.615974f, 2.270590f,
  2.089733f, 1.900703f, 0.817282f, 2.345506f, 2.728349f, 0.776276f, 1.882808f, 2.381747f,
  0.626826f, 1.698056f, 1.564135f, 2.069230f, 1.584012f, 2.318691f, 2.784085f, 2.321439f,
  1.287789f, 1.873939f, 1.938509f, 2.366931f, 1.497235f, 1.439013f, 2.785093f, 1.059509f,
  0.726349f, 1.391868f, 1.202786f, 2.121157f, 0.580812f, 0.735601f, 2.290780f, 1.109940f,
  1.589976f, 2.064378f, 1.775320f, 1.171153f, 2.335733f, 2.013564f, 0.552483f, 1.464142f,
  1.355975f, 2.324760f, 2.149520f, 0.827769f, 1.461377f, 2.261373f, 2.441865f, 2.260591f,
  1.025998f, 1.440526f, 0.747165f, 1.811700f, 1.935987f, 1.506869f, 1.040867f, 0.845263f,
  1.861470f, 2.299040f, 1.718228f, 0.567926f, 0.918545f, 2.086011f, 2.265755f, 2.004869f,
  1.853697f, 1.682213f, 0.633869f, 2.065978f, 2.345993f, 0.600010f, 1.668301f, 2.092115f,
  0.474524f, 1.490679f, 1.362270f, 1.838307f, 1.378443f, 2.048744f, 2.381730f, 2.041962f,
  1.085246f, 1.663136f, 1.718698f, 2.079838f, 1.294936f, 1.230819f, 2.378878f, 0.863940f,
};

// Feature stds from training (192 features)
static constexpr float kFeatStd[192] = {
  5.640917f, 5.850614f, 5.875398f, 5.810046f, 5.489097f, 5.639064f, 5.814719f, 5.857488f,
  5.907526f, 5.880238f, 5.901523f, 5.886196f, 5.809251f, 5.876165f, 5.483927f, 5.861886f,
  5.859579f, 5.848975f, 5.847031f, 5.700517f, 5.969014f, 5.831688f, 5.773424f, 5.802370f,
  5.835848f, 5.896096f, 5.635989f, 5.908012f, 5.886395f, 5.883692f, 5.851516f, 5.682363f,
  5.905924f, 5.822537f, 5.916232f, 5.488540f, 5.791875f, 5.873209f, 5.799204f, 5.918178f,
  5.921967f, 5.900486f, 5.515767f, 5.853590f, 5.858807f, 5.512128f, 5.891625f, 5.859546f,
  5.363428f, 5.917121f, 5.878615f, 5.887780f, 5.910470f, 5.877298f, 5.782053f, 5.871818f,
  5.876932f, 5.899826f, 5.895725f, 5.815412f, 5.954587f, 5.893755f, 5.804826f, 5.768018f,
  4.014396f, 4.074304f, 4.068505f, 4.256481f, 3.977277f, 4.003362f, 4.296505f, 4.053994f,
  4.095637f, 4.250304f, 4.178672f, 4.051883f, 4.362127f, 4.235273f, 3.949961f, 4.074847f,
  4.062206f, 4.348877f, 4.279499f, 3.981470f, 4.140448f, 4.310784f, 4.415397f, 4.311656f,
  4.058379f, 4.053221f, 4.004378f, 4.147777f, 4.189549f, 4.060062f, 4.050247f, 3.990761f,
  4.206835f, 4.325397f, 4.170592f, 3.984411f, 4.062205f, 4.274626f, 4.314578f, 4.252924f,
  4.165246f, 4.143425f, 3.985665f, 4.215228f, 4.356667f, 3.960316f, 4.106901f, 4.247245f,
  3.919852f, 4.096423f, 4.067114f, 4.175352f, 4.083612f, 4.233387f, 4.360720f, 4.254050f,
  4.065619f, 4.115298f, 4.134706f, 4.211278f, 4.121316f, 4.103688f, 4.365820f, 4.002383f,
  2.905401f, 3.235826f, 3.147075f, 3.574041f, 2.834859f, 2.900964f, 3.616836f, 3.076178f,
  3.334854f, 3.572155f, 3.460880f, 3.130896f, 3.644305f, 3.569870f, 2.861416f, 3.265129f,
  3.212042f, 3.647281f, 3.611663f, 2.922417f, 3.334342f, 3.624887f, 3.663806f, 3.623699f,
  3.038407f, 3.239315f, 2.897409f, 3.454749f, 3.517306f, 3.288007f, 3.062839f, 2.948911f,
  3.516137f, 3.625205f, 3.443416f, 2.831538f, 2.991084f, 3.583443f, 3.621285f, 3.562320f,
  3.473251f, 3.397612f, 2.835555f, 3.556777f, 3.655006f, 2.820974f, 3.371558f, 3.576294f,
  2.800675f, 3.290322f, 3.225279f, 3.492013f, 3.243338f, 3.570413f, 3.642924f, 3.567293f,
  3.090923f, 3.381319f, 3.419954f, 3.556603f, 3.216432f, 3.156761f, 3.638665f, 2.955197f,
};

// ============================================================
// Implementation
// ============================================================

template <typename T>
T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

FixedWeightDecoder::FixedWeightDecoder() : publish_rate_limiter_(kPublishRateSec) {
  // Copy training constants into member arrays
  for (size_t i = 0; i < kNumNeural; ++i) {
    channel_stds_[i] = kChannelStds[i];
  }
  for (size_t i = 0; i < kNumFeatures; ++i) {
    feat_mean_[i] = kFeatMean[i];
    feat_std_[i] = kFeatStd[i];
  }
}

void FixedWeightDecoder::initialize_thresholds() {
  // Compute per-channel thresholds: threshold[ch][t] = sigma[t] * channel_std[ch]
  for (size_t ch = 0; ch < kNumNeural; ++ch) {
    for (size_t t = 0; t < kNumThresholds; ++t) {
      channel_thresholds_[ch][t] = kSigmaThresholds[t] * channel_stds_[ch];
    }
  }
  spdlog::info("Initialized 3-threshold spike detection (3σ/4σ/5σ) for {} channels", kNumNeural);
}

bool FixedWeightDecoder::setup() {
  if (!get_app_config(
          [this](const synapse::ApplicationNodeConfig& configuration) {
            return validate_config(configuration);
          },
          application_config_)) {
    spdlog::error("Failed to get app config");
    return false;
  }

  if (!parse_config(application_config_)) {
    spdlog::error("Failed to parse app config");
    return false;
  }

  // Initialize spike thresholds from training channel stds
  initialize_thresholds();

  const uint32_t broadband_node_id = 1;
  if (!setup_reader(broadband_node_id)) {
    spdlog::warn("Failed to set up reader for controller");
    return false;
  }

  // Output tap: 7 values [joy_x, joy_y, rot, depth, lt, rt, gate]
  if (!create_tap<synapse::Tensor>("joystick_out")) {
    spdlog::warn("Failed to create tap for joystick out");
    return false;
  }

  if (enable_function_profiling_) {
    function_profiler_manager_.add("full_loop");
    function_profiler_manager_.add("inference");
    if (!enable_function_profiling(std::chrono::seconds(1))) {
      spdlog::error("Failed to enable function profile monitoring");
      return false;
    }
  }

  if (enable_inference_) {
    setup_inference();
  }

  return true;
}

void FixedWeightDecoder::main() {
  const float bin_size_ms = 10;
  std::vector<synapse::BroadbandFrame> broadband_frames;

  while (node_running_) {
    if (!wait_for_frames(broadband_frames, bin_size_ms)) {
      continue;
    }

    start_profile("full_loop");

    // Initialize filters on first frame
    const auto broadband_frame = broadband_frames.at(0);
    if (!filters_initialized_) {
      const size_t channel_count = broadband_frame.frame_data_size();
      const float sample_rate_hz = broadband_frame.sample_rate_hz();
      sample_rate_hz_ = sample_rate_hz;

      spdlog::info("Received first frames: {} channels at {} Hz", channel_count, sample_rate_hz);

      if (channel_count < kNumNeural) {
        spdlog::error("Expected at least {} neural channels, got {}", kNumNeural, channel_count);
        return;
      }

      initialize_filters(channel_count, sample_rate_hz, bin_size_ms);
      continue;
    }

    // Bandpass filter all channels for this bin
    const size_t n_channels = broadband_frames.at(0).frame_data_size();
    std::vector<std::vector<float>> filtered_channel_data(n_channels);
    for (auto& ch_data : filtered_channel_data) {
      ch_data.reserve(broadband_frames.size());
    }

    for (const auto& frame : broadband_frames) {
      const auto& frame_data = frame.frame_data();
      for (int ch = 0; ch < frame_data.size(); ++ch) {
        auto& filter = bandpass_filters_.at(ch);
        float filtered = filter->filter(frame_data[ch]);
        filtered_channel_data.at(ch).push_back(filtered);
      }
    }

    // Extract 192 features (3-threshold spike counts) from this bin
    auto features = extract_features(filtered_channel_data);

    // Normalize using training statistics
    normalize_features(features);

    // Add to circular buffer
    feature_buffer_.push_back(features);
    if (feature_buffer_.size() > kSeqLen) {
      feature_buffer_.pop_front();
    }

    // Default outputs: all zeros
    std::array<float, kNumOutputs> outputs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Run inference once we have a full window
    if (feature_buffer_.size() == kSeqLen && enable_inference_ && model_ && model_->is_ready()) {
      outputs = run_inference();
    }

    // Apply gate: if gate < threshold, suppress joystick axes
    float gate = outputs[6];
    if (gate < kGateThreshold) {
      outputs[0] = 0.0f;  // joy_x
      outputs[1] = 0.0f;  // joy_y
      outputs[2] = 0.0f;  // rot
      outputs[3] = 0.0f;  // depth
    }

    // Publish output tensor: 7 values
    synapse::Tensor output_tensor;
    const auto tensor_shape = {static_cast<int>(kNumOutputs)};
    output_tensor.mutable_shape()->Add(tensor_shape.begin(), tensor_shape.end());
    output_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
    output_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

    std::vector<float> output_vec(outputs.begin(), outputs.end());
    output_tensor.set_data(synapse::pack_tensor_data(output_vec));

    const auto current_time_ns = synapse::get_steady_clock_now();
    output_tensor.set_timestamp_ns(current_time_ns.count());

    if (publish_rate_limiter_.reset_if_elapsed()) {
      if (publish_tap("joystick_out", output_tensor)) {
        spdlog::info("Published: joy=[{:.3f},{:.3f}] rot={:.3f} depth={:.3f} trig=[{:.3f},{:.3f}] gate={:.3f}",
                     outputs[0], outputs[1], outputs[2], outputs[3],
                     outputs[4], outputs[5], outputs[6]);
      } else {
        spdlog::warn("Failed to publish tensor data");
      }
      stop_profile("full_loop");
      print_profile("full_loop");
    }
  }
}

std::array<float, kNumFeatures> FixedWeightDecoder::extract_features(
    const std::vector<std::vector<float>>& filtered_channel_data) {
  std::array<float, kNumFeatures> features = {};

  // For each neural channel, count threshold crossings at 3σ, 4σ, 5σ
  for (size_t ch = 0; ch < kNumNeural && ch < filtered_channel_data.size(); ++ch) {
    const auto& ch_data = filtered_channel_data[ch];

    for (size_t t = 0; t < kNumThresholds; ++t) {
      float threshold = channel_thresholds_[ch][t];
      uint32_t count = 0;

      for (const float sample : ch_data) {
        if (std::fabs(sample) > threshold) {
          count++;
        }
      }

      // Feature layout: [ch0_3σ, ch1_3σ, ..., ch63_3σ, ch0_4σ, ..., ch63_4σ, ch0_5σ, ..., ch63_5σ]
      features[t * kNumNeural + ch] = static_cast<float>(count);
    }
  }

  return features;
}

void FixedWeightDecoder::normalize_features(std::array<float, kNumFeatures>& features) {
  for (size_t i = 0; i < kNumFeatures; ++i) {
    features[i] = (features[i] - feat_mean_[i]) / (feat_std_[i] + 1e-8f);
  }
}

std::array<float, kNumOutputs> FixedWeightDecoder::run_inference() {
  std::array<float, kNumOutputs> outputs = {};

  auto inputs = model_->get_input_info();
  if (inputs.empty()) {
    return outputs;
  }

  // Build input tensor: shape (1, 192, 30) — features × time
  // Layout: for each feature f, then for each time step t
  std::vector<float> input_data(kNumFeatures * kSeqLen, 0.0f);

  for (size_t f = 0; f < kNumFeatures; ++f) {
    for (size_t t = 0; t < kSeqLen; ++t) {
      // input_data[f * kSeqLen + t] = feature_buffer_[t][f]
      input_data[f * kSeqLen + t] = feature_buffer_[t][f];
    }
  }

  start_profile("inference");
  auto result = model_->infer({input_data});
  stop_profile("inference");
  print_profile("inference");

  if (!result.success || result.outputs.empty()) {
    spdlog::warn("Inference failed");
    return outputs;
  }

  // Update benchmarking
  inference_count_++;
  inference_total_us_ += result.inference_time_us;
  inference_min_us_ = std::min(inference_min_us_, result.inference_time_us);
  inference_max_us_ = std::max(inference_max_us_, result.inference_time_us);

  if (inference_count_ % 100 == 0) {
    uint64_t avg_us = inference_total_us_ / inference_count_;
    spdlog::info("Inference stats: count={}, avg={} us, min={} us, max={} us",
                  inference_count_, avg_us, inference_min_us_, inference_max_us_);
  }

  // Parse 7 outputs: [joy_x, joy_y, rot, depth, lt, rt, gate]
  const auto& out = result.outputs[0];
  for (size_t i = 0; i < kNumOutputs && i < out.size(); ++i) {
    outputs[i] = out[i];
  }

  // Clamp joystick axes to [-1, 1]
  for (size_t i = 0; i < 4; ++i) {
    outputs[i] = clamp(outputs[i], -1.0f, 1.0f);
  }
  // Clamp triggers and gate to [0, 1]
  for (size_t i = 4; i < 7; ++i) {
    outputs[i] = clamp(outputs[i], 0.0f, 1.0f);
  }

  return outputs;
}

void FixedWeightDecoder::setup_inference() {
  auto runtimes = synapse::get_available_runtimes();
  spdlog::info("Available inference runtimes:");
  for (const auto& rt : runtimes) {
    const char* name = "unknown";
    switch (rt) {
      case synapse::InferenceRuntime::kCpu: name = "CPU (ONNX Runtime)"; break;
      case synapse::InferenceRuntime::kGpu: name = "GPU (QNN)"; break;
      case synapse::InferenceRuntime::kDsp: name = "DSP (QNN HTP)"; break;
      case synapse::InferenceRuntime::kAuto: name = "Auto"; break;
    }
    spdlog::info("  - {}", name);
  }

  model_ = synapse::create_model(model_name_);

  if (model_ && model_->is_ready()) {
    spdlog::info("Model '{}' loaded — input: (1, {}, {}), output: (1, {})",
                 model_name_, kNumFeatures, kSeqLen, kNumOutputs);

    auto inputs = model_->get_input_info();
    for (const auto& input : inputs) {
      std::string shape_str;
      for (size_t i = 0; i < input.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(input.shape[i]);
      }
      spdlog::info("  Input: {} shape=[{}] elements={}", input.name, shape_str, input.element_count);
    }

    auto model_outputs = model_->get_output_info();
    for (const auto& output : model_outputs) {
      std::string shape_str;
      for (size_t i = 0; i < output.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(output.shape[i]);
      }
      spdlog::info("  Output: {} shape=[{}] elements={}", output.name, shape_str, output.element_count);
    }
  } else {
    spdlog::warn("Model '{}' not available — deploy with: synapsectl deploy-model decoder.onnx --name {} -u <device>",
                  model_name_, model_name_);
  }
}

bool FixedWeightDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                                         float bin_size_ms) {
  if (bin_size_ms <= 0) {
    spdlog::warn("invalid bin size of: {}", bin_size_ms);
    return false;
  }

  const float bin_size_sec = bin_size_ms / 1000;
  const size_t target_num_of_frames = bin_size_sec * sample_rate_hz_;

  frames.clear();

  while (node_running_) {
    auto messages = data_reader_->receive_multipart();
    if (messages.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    frames.reserve(frames.size() + messages.size());

    for (auto& message : messages) {
      const auto maybe_frame =
          synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
      if (!maybe_frame.has_value()) {
        spdlog::warn("Failed to parse broadband frame");
        if (frames.empty()) {
          return false;
        }
        return true;
      }

      const auto& broadband_frame = maybe_frame.value();

      const auto dropped_frames =
          detect_dropped_frames(last_sequence_number_, broadband_frame.sequence_number());
      if (dropped_frames != 0) {
        spdlog::warn("Dropped: {} frames", dropped_frames);
      }
      last_sequence_number_ = broadband_frame.sequence_number();

      frames.push_back(broadband_frame);
    }

    if (frames.size() >= target_num_of_frames) {
      return true;
    }
  }
  return false;
}

int FixedWeightDecoder::detect_dropped_frames(const uint64_t last_sequence_number,
                                              const uint64_t current_sequence_number) {
  const auto expected_sequence_number = last_sequence_number + 1;
  return (current_sequence_number - expected_sequence_number);
}

void FixedWeightDecoder::initialize_filters(const size_t channel_count, const float sample_rate_hz,
                                            const float bin_size_ms) {
  spdlog::info("Initializing: sample_rate={} Hz, channels={}, bin_size={} ms",
               sample_rate_hz, channel_count, bin_size_ms);

  bandpass_filters_.clear();
  bandpass_filters_.reserve(channel_count);
  for (size_t ch = 0; ch < channel_count; ++ch) {
    auto filter_ptr = synapse::create_bandpass_filter<kSpectralFilterOrder>(
        sample_rate_hz, low_cutoff_hz_, high_cutoff_hz_);
    if (filter_ptr == nullptr) {
      spdlog::error("Failed to create filter for channel: {}", ch);
    }
    bandpass_filters_.push_back(std::move(filter_ptr));
  }
  spdlog::info("Initialized {} bandpass filters ({}-{} Hz)", channel_count, low_cutoff_hz_, high_cutoff_hz_);
  filters_initialized_ = true;
}

bool FixedWeightDecoder::validate_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& parameters = configuration.parameters();

  const std::vector<std::string> required = {
    "low_cutoff_hz", "high_cutoff_hz", "enable_function_profiling"
  };

  for (const auto& key : required) {
    if (!parameters.contains(key)) {
      spdlog::error("{} not found in configuration", key);
      return false;
    }
  }

  return true;
}

bool FixedWeightDecoder::parse_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& parameters = configuration.parameters();
  try {
    low_cutoff_hz_ = parameters.at("low_cutoff_hz").number_value();
    high_cutoff_hz_ = parameters.at("high_cutoff_hz").number_value();
    enable_function_profiling_ = parameters.at("enable_function_profiling").bool_value();

    if (parameters.contains("enable_inference")) {
      enable_inference_ = parameters.at("enable_inference").bool_value();
    }
    if (parameters.contains("model_name")) {
      model_name_ = parameters.at("model_name").string_value();
    }

    spdlog::info("Config: filter={}-{} Hz, inference={}, model={}",
                 low_cutoff_hz_, high_cutoff_hz_, enable_inference_, model_name_);
    application_config_ = configuration;
    return true;
  } catch (const std::exception& e) {
    spdlog::error("Failed to parse configuration: {}", e.what());
    return false;
  }
}

}  // namespace app

int main(const int, const char**) { return synapse::Entrypoint<app::FixedWeightDecoder>(); }
