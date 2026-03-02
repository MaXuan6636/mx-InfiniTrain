#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

// MNIST dataset loader and preprocessing wrapper.
class MNISTDataset : public infini_train::Dataset {
public:
    // Data type codes used by SN3 Pascal Vincent files.
    enum class SN3PascalVincentType : int {
        kUINT8,
        kINT8,
        kINT16,
        kINT32,
        kFLOAT32,
        kFLOAT64,
        kINVALID,
    };

    // Parsed SN3 file content.
    struct SN3PascalVincentFile {
        SN3PascalVincentType type = SN3PascalVincentType::kINVALID;
        std::vector<int64_t> dims;
        infini_train::Tensor tensor;
    };

    // @param prefix Dataset directory, e.g. /path/to/mnist
    // @param train true for train-* files, false for t10k-* files
    MNISTDataset(const std::string &prefix, bool train);

    // Returns the idx-th sample pair: (image, label).
    std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    operator[](size_t idx) const override;

    // Number of samples in the dataset.
    size_t Size() const override;

private:
    // Parsed image and label files.
    SN3PascalVincentFile image_file_;
    SN3PascalVincentFile label_file_;
    // Per-sample dims (without batch dim).
    std::vector<int64_t> image_dims_;
    std::vector<int64_t> label_dims_;
};
