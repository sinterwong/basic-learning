#include "fbank.hpp"
#include <algorithm>
#include <cstring>
#include <kissfft/kiss_fft.h>
#include <kissfft/kiss_fftr.h>
#include <stdexcept>

FbankComputer::FbankComputer(const FbankOptions &opts)
    : opts_(opts), rng_(std::random_device{}()), normal_dist_(0.0f, 1.0f) {

  // Validate parameters
  if (opts_.frame_length <= 0 || opts_.frame_shift <= 0) {
    throw std::invalid_argument("Frame length and shift must be positive");
  }
  if (opts_.sample_frequency <= 0) {
    throw std::invalid_argument("Sample frequency must be positive");
  }
  if (opts_.num_mel_bins <= 0) {
    throw std::invalid_argument("Number of mel bins must be positive");
  }

  // Calculate frame parameters
  frame_length_samples_ =
      static_cast<int>(opts_.frame_length * kMsToSec * opts_.sample_frequency);
  frame_shift_samples_ =
      static_cast<int>(opts_.frame_shift * kMsToSec * opts_.sample_frequency);
  padded_window_size_ = opts_.round_to_power_of_two
                            ? GetNextPowerOfTwo(frame_length_samples_)
                            : frame_length_samples_;

  if (padded_window_size_ % 2 != 0) {
    throw std::runtime_error("Padded window size must be even");
  }

  // Initialize FFT
  kiss_fftr_cfg cfg =
      kiss_fftr_alloc(padded_window_size_, false, nullptr, nullptr);
  if (!cfg) {
    throw std::runtime_error("Failed to allocate FFT config");
  }
  fft_config_.reset(cfg);
  if (!fft_config_) {
    throw std::runtime_error("Failed to allocate FFT config");
  }

  // Initialize mel banks
  auto mel_banks_pair = GetMelBanks();
  mel_banks_ = std::move(mel_banks_pair.first);
}

FbankComputer::~FbankComputer() = default;

float FbankComputer::MelScale(float freq) {
  return 1127.0f * std::log(1.0f + freq / 700.0f);
}

float FbankComputer::InverseMelScale(float mel_freq) {
  return 700.0f * (std::exp(mel_freq / 1127.0f) - 1.0f);
}

float FbankComputer::VtlnWarpFreq(float vtln_low, float vtln_high,
                                  float low_freq, float high_freq,
                                  float vtln_warp_factor, float freq) {
  // Implementation matching PyTorch's vtln_warp_freq
  float scale = 1.0f / vtln_warp_factor;
  float f_low = vtln_low * std::max(1.0f, vtln_warp_factor);
  float f_high = vtln_high * std::min(1.0f, vtln_warp_factor);
  float scale_left = (f_low * scale - low_freq) / (f_low - low_freq);
  float scale_right = (high_freq - f_high * scale) / (high_freq - f_high);

  if (freq < low_freq || freq > high_freq)
    return freq;
  if (freq < f_low)
    return low_freq + scale_left * (freq - low_freq);
  if (freq < f_high)
    return freq * scale;
  return high_freq + scale_right * (freq - high_freq);
}

float FbankComputer::VtlnWarpMelFreq(float vtln_low, float vtln_high,
                                     float low_freq, float high_freq,
                                     float vtln_warp_factor, float mel_freq) {
  return MelScale(VtlnWarpFreq(vtln_low, vtln_high, low_freq, high_freq,
                               vtln_warp_factor, InverseMelScale(mel_freq)));
}

void FbankComputer::PreprocessFrame(std::vector<float> &frame,
                                    float *log_energy) {
  // Compute raw energy if needed
  if (opts_.raw_energy && log_energy) {
    *log_energy = GetLogEnergy(frame);
  }

  // Add dither
  if (opts_.dither != 0.0f) {
    for (auto &sample : frame) {
      sample += opts_.dither * normal_dist_(rng_);
    }
  }

  // Remove DC offset
  if (opts_.remove_dc_offset) {
    float mean = 0.0f;
    for (float sample : frame) {
      mean += sample;
    }
    mean /= frame.size();
    for (auto &sample : frame) {
      sample -= mean;
    }
  }

  // Preemphasis
  if (opts_.preemphasis_coefficient != 0.0f) {
    std::vector<float> preemphasized(frame.size());
    preemphasized[0] = frame[0] * (1.0f - opts_.preemphasis_coefficient);
    for (size_t i = 1; i < frame.size(); i++) {
      preemphasized[i] =
          frame[i] - opts_.preemphasis_coefficient * frame[i - 1];
    }
    frame = std::move(preemphasized);
  }

  // Compute non-raw energy if needed
  if (!opts_.raw_energy && log_energy) {
    *log_energy = GetLogEnergy(frame);
  }
}

std::vector<std::vector<float>>
FbankComputer::Compute(const std::vector<float> &waveform) {
  auto frames = GetStridedFrames(waveform);
  std::vector<std::vector<float>> features;
  features.reserve(frames.size());

  std::vector<kiss_fft_cpx> fft_out((padded_window_size_ / 2) + 1);
  std::vector<float> padded_frame(padded_window_size_);
  auto window = GetWindowFunction(frame_length_samples_);

  for (auto &frame : frames) {
    float log_energy = 0.0f;
    PreprocessFrame(frame, opts_.use_energy ? &log_energy : nullptr);

    // Apply window function
    for (int i = 0; i < frame_length_samples_; i++) {
      frame[i] *= window[i];
    }

    // Prepare FFT input
    std::copy(frame.begin(), frame.end(), padded_frame.begin());
    std::fill(padded_frame.begin() + frame_length_samples_, padded_frame.end(),
              0.0f);

    // Perform FFT
    kiss_fftr(fft_config_.get(), padded_frame.data(),
              reinterpret_cast<kiss_fft_cpx *>(fft_out.data()));

    // Compute power spectrum
    std::vector<float> power_spectrum((padded_window_size_ / 2) + 1);
    for (size_t i = 0; i < power_spectrum.size(); i++) {
      float re = fft_out[i].r;
      float im = fft_out[i].i;
      power_spectrum[i] = re * re + im * im;

      if (opts_.use_power) {
        power_spectrum[i] = std::max(power_spectrum[i], kEpsilon);
      } else {
        power_spectrum[i] = std::sqrt(power_spectrum[i]);
      }
    }

    // Apply mel filterbanks
    std::vector<float> mel_energies(opts_.num_mel_bins);
    for (int i = 0; i < opts_.num_mel_bins; i++) {
      float energy = 0.0f;
      for (size_t j = 0; j < power_spectrum.size(); j++) {
        energy += mel_banks_[i][j] * power_spectrum[j];
      }

      if (opts_.use_log_fbank) {
        energy = std::log(std::max(energy, kEpsilon));
      }
      mel_energies[i] = energy;
    }

    // Add energy if requested
    if (opts_.use_energy) {
      if (opts_.htk_compat) {
        mel_energies.push_back(log_energy);
      } else {
        mel_energies.insert(mel_energies.begin(), log_energy);
      }
    }

    features.push_back(std::move(mel_energies));
  }

  // Subtract mean if requested
  if (opts_.subtract_mean && !features.empty()) {
    size_t feat_dim = features[0].size();
    std::vector<float> means(feat_dim, 0.0f);

    // Compute means
    for (const auto &feature : features) {
      for (size_t i = 0; i < feat_dim; i++) {
        means[i] += feature[i];
      }
    }
    for (auto &mean : means) {
      mean /= features.size();
    }

    // Subtract means
    for (auto &feature : features) {
      for (size_t i = 0; i < feat_dim; i++) {
        feature[i] -= means[i];
      }
    }
  }

  return features;
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
FbankComputer::GetMelBanks() const {
  float nyquist = opts_.sample_frequency / 2.0f;
  float high_freq =
      opts_.high_freq > 0.0f ? opts_.high_freq : nyquist + opts_.high_freq;
  float low_freq = opts_.low_freq;

  if (low_freq < 0.0f || low_freq >= nyquist || high_freq <= 0.0f ||
      high_freq > nyquist || low_freq >= high_freq) {
    throw std::invalid_argument("Invalid frequency range");
  }

  float mel_low = MelScale(low_freq);
  float mel_high = MelScale(high_freq);
  float mel_freq_delta = (mel_high - mel_low) / (opts_.num_mel_bins + 1);

  std::vector<float> center_freqs(opts_.num_mel_bins);
  std::vector<std::vector<float>> mel_filters(
      opts_.num_mel_bins, std::vector<float>(padded_window_size_ / 2 + 1));

  // Calculate FFT bin frequencies
  std::vector<float> fft_freqs(padded_window_size_ / 2 + 1);
  for (size_t i = 0; i < fft_freqs.size(); i++) {
    fft_freqs[i] = i * opts_.sample_frequency / padded_window_size_;
  }

  for (int i = 0; i < opts_.num_mel_bins; i++) {
    float left_mel = mel_low + i * mel_freq_delta;
    float center_mel = mel_low + (i + 1) * mel_freq_delta;
    float right_mel = mel_low + (i + 2) * mel_freq_delta;

    // Apply VTLN warping if needed
    if (opts_.vtln_warp != 1.0f) {
      left_mel =
          VtlnWarpMelFreq(opts_.vtln_low, opts_.vtln_high, opts_.low_freq,
                          high_freq, opts_.vtln_warp, left_mel);
      center_mel =
          VtlnWarpMelFreq(opts_.vtln_low, opts_.vtln_high, opts_.low_freq,
                          high_freq, opts_.vtln_warp, center_mel);
      right_mel =
          VtlnWarpMelFreq(opts_.vtln_low, opts_.vtln_high, opts_.low_freq,
                          high_freq, opts_.vtln_warp, right_mel);
    }

    center_freqs[i] = InverseMelScale(center_mel);

    // Create triangular filters
    for (size_t j = 0; j < fft_freqs.size(); j++) {
      float mel = MelScale(fft_freqs[j]);
      float weight = 0.0f;

      if (mel > left_mel && mel < right_mel) {
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel - mel) / (right_mel - center_mel);
        }
      }
      mel_filters[i][j] = weight;
    }
  }

  return {mel_filters, center_freqs};
}

std::vector<std::vector<float>>
FbankComputer::GetStridedFrames(const std::vector<float> &waveform) const {
  std::vector<std::vector<float>> frames;
  int num_samples = waveform.size();

  if (num_samples == 0) {
    return frames;
  }

  int num_frames;
  if (opts_.snip_edges) {
    if (num_samples < frame_length_samples_) {
      return frames;
    }
    num_frames =
        1 + (num_samples - frame_length_samples_) / frame_shift_samples_;
  } else {
    num_frames =
        (num_samples + frame_shift_samples_ / 2) / frame_shift_samples_;
  }

  frames.reserve(num_frames);

  for (int i = 0; i < num_frames; i++) {
    std::vector<float> frame(frame_length_samples_);
    int start_sample = i * frame_shift_samples_;

    if (opts_.snip_edges) {
      // Just copy the samples directly
      for (int j = 0; j < frame_length_samples_; j++) {
        frame[j] = waveform[start_sample + j];
      }
    } else {
      // Handle edge effects by reflection
      for (int j = 0; j < frame_length_samples_; j++) {
        int sample_index = start_sample + j - frame_length_samples_ / 2;

        if (sample_index < 0) {
          // Reflect across the beginning
          sample_index = -sample_index - 1;
        } else if (sample_index >= num_samples) {
          // Reflect across the end
          sample_index = 2 * num_samples - sample_index - 1;
        }

        frame[j] = waveform[sample_index];
      }
    }

    frames.push_back(std::move(frame));
  }

  return frames;
}

float FbankComputer::GetLogEnergy(const std::vector<float> &frame) const {
  float energy = 0.0f;
  for (float sample : frame) {
    energy += sample * sample;
  }

  energy = std::max(energy, kEpsilon);
  if (opts_.energy_floor > 0.0f) {
    energy = std::max(energy, opts_.energy_floor);
  }

  return std::log(energy);
}

// Window function implementations
std::vector<float> FbankComputer::GetWindowFunction(int size) const {
  if (opts_.window_type == "hamming") {
    return HammingWindow(size);
  } else if (opts_.window_type == "hanning") {
    return HanningWindow(size);
  } else if (opts_.window_type == "povey") {
    return PoveyWindow(size);
  } else if (opts_.window_type == "rectangular") {
    return std::vector<float>(size, 1.0f);
  } else if (opts_.window_type == "blackman") {
    return BlackmanWindow(size);
  } else {
    throw std::runtime_error("Unsupported window type: " + opts_.window_type);
  }
}

std::vector<float> FbankComputer::HammingWindow(int size) const {
  std::vector<float> window(size);
  for (int i = 0; i < size; i++) {
    window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
  }
  return window;
}

std::vector<float> FbankComputer::HanningWindow(int size) const {
  std::vector<float> window(size);
  for (int i = 0; i < size; i++) {
    window[i] = 0.5f - 0.5f * std::cos(2.0f * M_PI * i / (size - 1));
  }
  return window;
}

std::vector<float> FbankComputer::BlackmanWindow(int size) const {
  std::vector<float> window(size);
  float a = 2.0f * M_PI / (size - 1);
  for (int i = 0; i < size; i++) {
    window[i] = opts_.blackman_coeff - 0.5f * std::cos(a * i) +
                (0.5f - opts_.blackman_coeff) * std::cos(2.0f * a * i);
  }
  return window;
}

std::vector<float> FbankComputer::PoveyWindow(int size) const {
  auto window = HanningWindow(size);
  for (auto &x : window) {
    x = std::pow(x, 0.85f);
  }
  return window;
}

int FbankComputer::GetNextPowerOfTwo(int x) {
  if (x <= 1)
    return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}