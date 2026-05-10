// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_DEVICE_MANAGER_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_DEVICE_MANAGER_H_

#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

// Keeps track of a bunch of VPNet models, intended to be one per device, and
// gives them out based on usage. When you request a device you specify how much
// work you're going to give it, which is assumed done once the loan is
// returned.
class DeviceManager {
 public:
  DeviceManager() {
    learning_ = false;
    multiple_devices_ = false;
  }

  void AddDevice(VPNetModel model) {  // Not thread safe.
    devices.push_back(std::make_unique<Device>(std::move(model)));
    multiple_devices_ = devices.size() > 1;
  }

  // Acts as a pointer to the model, but lets the manager know when you're done.
  class DeviceLoan {
   public:
    // DeviceLoan is not public constructible and is move only.
    DeviceLoan(DeviceLoan&& other) noexcept
        : manager_(other.manager_),
          model_(other.model_),
          model_mu_(other.model_mu_),
          device_id_(other.device_id_),
          requests_(other.requests_) {
      other.manager_ = nullptr;
      other.model_ = nullptr;
      other.model_mu_ = nullptr;
      other.device_id_ = -1;
      other.requests_ = 0;
    }
    DeviceLoan& operator=(DeviceLoan&& other) noexcept {
      if (this != &other) {
        Release();
        manager_ = other.manager_;
        model_ = other.model_;
        model_mu_ = other.model_mu_;
        device_id_ = other.device_id_;
        requests_ = other.requests_;
        other.manager_ = nullptr;
        other.model_ = nullptr;
        other.model_mu_ = nullptr;
        other.device_id_ = -1;
        other.requests_ = 0;
      }
      return *this;
    }
    DeviceLoan(const DeviceLoan&) = delete;
    DeviceLoan& operator=(const DeviceLoan&) = delete;

    ~DeviceLoan() { Release(); }
    VPNetModel* operator->() { return model_; }

   private:
    DeviceLoan(DeviceManager* manager, VPNetModel* model,
               absl::Mutex* model_mu, int device_id, int requests)
        : manager_(manager),
          model_(model),
          model_mu_(model_mu),
          device_id_(device_id),
          requests_(requests) {}
    void Release() {
      if (model_mu_ != nullptr) {
        model_mu_->Unlock();
        model_mu_ = nullptr;
      }
      if (manager_ != nullptr) {
        manager_->Return(device_id_, requests_);
        manager_ = nullptr;
      }
    }
    DeviceManager* manager_;
    VPNetModel* model_;
    absl::Mutex* model_mu_;
    int device_id_;
    int requests_;
    friend DeviceManager;
  };

  // Gives the device with the fewest outstanding requests.
  DeviceLoan Get(int requests, int device_id = -1) {
    VPNetModel* model = nullptr;
    absl::Mutex* model_mu = nullptr;
    {
      absl::MutexLock lock(m_);
      if (device_id < 0) {
        // The starting device changes depending on if we are allowed to
        // use the first device or not.
        device_id = 0 + (learning_ && multiple_devices_);
        for (int i = 1 + (learning_ && multiple_devices_); i < devices.size();
             ++i) {
          if (devices[i]->requests < devices[device_id]->requests) {
            device_id = i;
          }
        }
      }
      devices[device_id]->requests += requests;
      model = &devices[device_id]->model;
      model_mu = &devices[device_id]->model_mu;
    }
    model_mu->Lock();
    return DeviceLoan(this, model, model_mu, device_id, requests);
  }

  // A member to ensure that when device:0 is learning and there are
  // multiple devices available, that device:0 does not take on any
  // inference requests from the actors and evaluators. These inference
  // requests should be dealt with by the other available devices.
  void SetLearning(bool value) {
    absl::MutexLock lock(m_);
    learning_ = value;
  }

  int Count() const { return devices.size(); }

 private:
  void Return(int device_id, int requests) {
    absl::MutexLock lock(m_);
    devices[device_id]->requests -= requests;
  }

  struct Device {
    explicit Device(VPNetModel model) : model(std::move(model)) {}
    VPNetModel model;
    int requests = 0;
    absl::Mutex model_mu;
  };

  bool learning_;
  bool multiple_devices_;
  std::vector<std::unique_ptr<Device>> devices;
  absl::Mutex m_;
};

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_DEVICE_MANAGER_H_
