#pragma once
#include "kiss_fftr.h"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

class FbankComputer {
public:
  struct FbankOptions {
    float blackman_coeff = 0.42f;
    int channel = -1;
    float dither = 0.0f;
    float energy_floor = 1.0f;
    float frame_length = 25.0f;
    float frame_shift = 10.0f;
    float high_freq = 0.0f;
    bool htk_compat = false;
    float low_freq = 20.0f;
    int num_mel_bins = 23;
    float preemphasis_coefficient = 0.97f;
    bool raw_energy = true;
    bool remove_dc_offset = true;
    bool round_to_power_of_two = true;
    float sample_frequency = 16000.0f;
    bool snip_edges = true;
    bool subtract_mean = false;
    bool use_energy = false;
    bool use_log_fbank = true;
    bool use_power = true;
    float vtln_high = -500.0f;
    float vtln_low = 100.0f;
    float vtln_warp = 1.0f;
    std::string window_type = "povey";
  };

  explicit FbankComputer(const FbankOptions &opts);
  ~FbankComputer();

  // Main compute function
  std::vector<std::vector<float>> Compute(const std::vector<float> &waveform);

private:
  static constexpr float kEpsilon = 1.1920928955078125e-07f;
  static constexpr float kMsToSec = 0.001f;

  struct KissFFTRDeleter {
    void operator()(kiss_fftr_cfg cfg) { kiss_fftr_free(cfg); }
  };

  // Utility functions
  static int GetNextPowerOfTwo(int x);
  static float MelScale(float freq);
  static float InverseMelScale(float mel_freq);
  static float VtlnWarpFreq(float vtln_low, float vtln_high, float low_freq,
                            float high_freq, float vtln_warp_factor,
                            float freq);
  static float VtlnWarpMelFreq(float vtln_low, float vtln_high, float low_freq,
                               float high_freq, float vtln_warp_factor,
                               float mel_freq);

  // Window functions
  std::vector<float> GetWindowFunction(int size) const;
  std::vector<float> HammingWindow(int size) const;
  std::vector<float> HanningWindow(int size) const;
  std::vector<float> BlackmanWindow(int size) const;
  std::vector<float> PoveyWindow(int size) const;

  // Processing functions
  std::vector<std::vector<float>>
  GetStridedFrames(const std::vector<float> &waveform) const;
  float GetLogEnergy(const std::vector<float> &frame) const;
  void PreprocessFrame(std::vector<float> &frame, float *log_energy = nullptr);
  std::pair<std::vector<std::vector<float>>, std::vector<float>>
  GetMelBanks() const;

  // Member variables
  FbankOptions opts_;
  int frame_length_samples_;
  int frame_shift_samples_;
  int padded_window_size_;
  std::unique_ptr<kiss_fftr_state, KissFFTRDeleter> fft_config_;
  std::vector<std::vector<float>> mel_banks_;
  std::mt19937 rng_; // Random number generator
  std::normal_distribution<float> normal_dist_;
};