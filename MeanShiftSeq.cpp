//
// Created by gioste.
//

#include "MeanShiftSeq.h"
#include <cmath>
#include "Utils.h"

// NB std::vector<Vec3f> data; è un Array di strutture (AoS) fornito da openCV

// Implementazione del Mean Shift Clustering Sequenziale
void meanShift_seq(const std::vector<cv::Vec3f>& data, std::vector<cv::Vec3f>& modes, float bandwidth, float epsilon) {
    // Inizializza 'modes' con i dati originali (ogni punto è inizialmente il proprio centro)
    modes = data;

    // Calcola il quadrato della bandwidth per evitare di fare la radice quadrata ad ogni iterazione
    const float bandwidthSquared = bandwidth * bandwidth;

    // Itera su tutti i punti dell'insieme dei dati
    for (size_t i = 0; i < modes.size(); ++i) {
        cv::Vec3f point = modes[i]; // Punto di partenza per l'iterazione del Mean Shift
        cv::Vec3f shift; // Vettore per accumulare lo spostamento del punto

        do {
            shift = cv::Vec3f(0, 0, 0); // Reset dello spostamento ad ogni iterazione
            float totalWeight = 0.0f;    // Variabile per sommare i pesi

            // Ciclo per calcolare il nuovo centro di massa ponderato (itera su tutti i punti)
            for (const auto& otherPoint : data) {
                float distSquared = squaredDistance(point, otherPoint); // Distanza euclidea al quadrato tra punto corrente e punto di riferimento

                // Applica kernel gaussiano per calcolare il peso del contributo
                if (distSquared < bandwidthSquared) { // Controlla se il punto rientra nella finestra di bandwidth
                    float weight = exp(-distSquared / (2 * bandwidthSquared)); // KERNEL GAUSSIANO
                    shift += weight * otherPoint; // Accumula il contributo ponderato del punto vicino
                    totalWeight += weight; // Somma il peso
                }
            }

            // Normalizza il nuovo punto (media pesata)
            if (totalWeight > 0) {
                shift /= totalWeight;
            }

            // Calcola la distanza di spostamento (distanza tra vecchia e nuova posizione)
            float shiftDistance = std::sqrt(squaredDistance(point, shift));
            point = shift; // Aggiorna il punto con la nuova posizione calcolata

            // Condizione di arresto: se lo spostamento è inferiore alla soglia epsilon, esci dal ciclo (CONDIZIONE CONVERGENZA SODDISFATTA)
            if (shiftDistance < epsilon) break;

        } while (true);

        // Aggiorna punto finale nel vettore modes
        modes[i] = point;
    }
}

// Segmentazione immagine con Mean Shift Sequenziale
cv::Mat segmentImage_seq(const cv::Mat& image, float bandwidth, float epsilon) {
    // Converte immagine da formato OpenCV (cv::Mat) a un vettore di punti RGB
    std::vector<cv::Vec3f> data;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Estrae il valore RGB del pixel in posizione (x, y)
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);

            // Converte i valori RGB da uchar (0-255) a float e li memorizza in data
            data.push_back(cv::Vec3f(color[0], color[1], color[2])); // Usa solo i canali RGB
        }
    }

    // Applica il Mean Shift sequenziale
    std::vector<cv::Vec3f> modes; // Vettore per memorizzare i risultati della segmentazione dopo Mean Shift
    meanShift_seq(data, modes, bandwidth, epsilon); // Applica Mean Shift sequenziale a lista di punti RGB

    // Riorganizziamo i risultati in una nuova immagine
    cv::Mat result = cv::Mat::zeros(image.size(), image.type()); // Crea un'immagine vuota con le stesse dimensioni dell'input

    // Riempie l'immagine con i risultati della segmentazione
    for (size_t i = 0; i < data.size(); ++i) {
        // Ricostruisce le coordinate x e y del pixel a partire da indice vettore
        int x = static_cast<int>(i % image.cols); // Calcola la posizione x
        int y = static_cast<int>(i / image.cols); // Calcola la posizione y

        // Converte i valori float (Mean Shift output) in uchar (0-255) per OpenCV
        cv::Vec3b color = cv::Vec3b(
            static_cast<uchar>(modes[i][0]),
            static_cast<uchar>(modes[i][1]),
            static_cast<uchar>(modes[i][2])
        );

        // Assegna colore segmentato a pixel corrispondente in immagine risultante
        result.at<cv::Vec3b>(y, x) = color;
    }

    return result; // Restituisce immagine segmentata
}
