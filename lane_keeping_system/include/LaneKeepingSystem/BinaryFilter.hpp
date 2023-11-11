// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file PIDController.hpp
 * @author Jongrok Lee (lrrghdrh@naver.com)
 * @author Jiho Han
 * @author Haeryong Lim
 * @author Chihyeon Lee
 * @brief PID Controller Class header file
 * @version 1.1
 * @date 2023-05-02
 */
#ifndef BINARY_FILTER_HPP_
#define BINARY_FILTER_HPP_

#include <deque>
#include <iostream>
#include <memory>
#include <vector>

namespace Xycar {
/**
 * @brief PID Controller Class
 * @tparam PREC Precision of data
 */
template <typename PREC>
class BinaryFilter
{
public:
    using Ptr = std::unique_ptr<BinaryFilter>; ///< Pointer type of this class

    /**
     * @brief Construct a new BinaryFilter
     *
     * @param[in] sampleSize 모집할 표본 사이즈
     * @param[in] prior 성공할 확률
     */
    BinaryFilter(uint32_t sampleSize, PREC prior);

    /**
     * @brief Add new data to filter
     *
     * @param[in] newSample New position to be used in filtering
     */
    void addSample(bool newSample);

    /**
     * @brief Get the filtered data
     *
     * @return 대상의 확률을 반환한다.
     */
    const PREC getResult() const { return mFilteringResult; }

private:
    const PREC mPrior;
    const PREC mSampleSize;

    PREC mFilteringResult;

    std::deque<bool> mSamples; ///< Deque including values of samples

    PREC update(uint32_t sampleSize);
};
} // namespace Xycar
#endif // BINARY_FILTER_HPP_
