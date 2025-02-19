//
// Created by gioste.
//

#include "Utils.h"

//#pragma omp simd // Direttiva che si aggiunta per vettorizzare
// Funzione per calcolare la distanza euclidea al quadrato sui canali RGB
float squaredDistance(const cv::Vec3f& p1, const cv::Vec3f& p2) {
    float dist = 0.0f;
    for (int i = 0; i < 3; ++i) {
        dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return dist;
}