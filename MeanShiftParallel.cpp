//
// Created by gioste.
//

#include "MeanShiftParallel.h"
#include <omp.h>
#include <cmath>
#include "Utils.h"

// NB struct SoAData; la seguente struttura dati è una Structure of Arrays (SoA)

// Converte i dati dell'immagine da AoS (Array of Structures) a SoA (Structure of Arrays)
SoAData convertToSoA(const cv::Mat& image) {
    SoAData data; // Struttura SoA per memorizzare i canali RGB separatamente

    // Riserva memoria per evitare riallocazioni dinamiche durante l'inserimento
    data.r.reserve(image.rows * image.cols);
    data.g.reserve(image.rows * image.cols);
    data.b.reserve(image.rows * image.cols);

    // Scansiona ogni pixel dell'immagine
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);  // Estrae il valore RGB del pixel (Array of Structures - AoS)

            // Inserisce i singoli canali di colore nei rispettivi vettori SoA
            data.r.push_back(static_cast<float>(color[0])); // Rosso
            data.g.push_back(static_cast<float>(color[1])); // Verde
            data.b.push_back(static_cast<float>(color[2])); // Blu
        }
    }
    return data; // Restituisce la struttura SoA contenente i dati RGB separati
}

// Implementazione parallela di Mean Shift (lavora con SoA)
void meanShift_parallel(const SoAData& data, SoAData& modes, float bandwidth, float epsilon) {
    // Assicura che modes abbia la stessa dimensione di data
    modes.r.resize(data.r.size());
    modes.g.resize(data.g.size());
    modes.b.resize(data.b.size());

    // First Touch: inizializzza modes per ogni thread con barriere implicite (copia manuale in un ciclo parallelo per rispettare il First Touch)
    #pragma omp parallel for // Distribuisce i dati tra le cache locali dei thread senza attendere una sincronizzazione alla fine.
    for (size_t i = 0; i < data.r.size(); ++i) {
        modes.r[i] = data.r[i];
        modes.g[i] = data.g[i];
        modes.b[i] = data.b[i];
    }

    const float bandwidthSquared = bandwidth * bandwidth; // Precalcolo banda al quadrato per ottimizzare i calcoli

    // Parallelizza il ciclo principale (per calcolo Mean Shit)
    #pragma omp parallel // Inizia una regione parallela
    {
        #pragma omp for schedule(dynamic, 64) nowait // divide cicli per i threads del team, ovvero parallelizza con OpenMP, leva barriere implicite (nowait), usa blocchi di 64 elementi per distribuire il carico in modo più equo tra i thread
        for (size_t i = 0; i < modes.r.size(); ++i) {
            float pointR = modes.r[i], pointG = modes.g[i], pointB = modes.b[i]; // Inizializza il punto di riferimento per il Mean Shift

            // Loop per spostare il punto fino alla convergenza
            do {
                // Variabili locali per ridurre la sincronizzazione (servono per calcolo nuova posizone)
                float localShiftR = 0.0f, localShiftG = 0.0f, localShiftB = 0.0f, localTotalWeight = 0.0f;

                // Ciclo interno per calcolare contributo di ogni punto (provato a parallelizzare e vettorizzazione esplicita, ma rallentano le prestazioni)
                // ! Non è parallelizzabile !
                for (size_t j = 0; j < data.r.size(); ++j) {
                    // Calcola distanza euclidea al quadrato tra punto corrente e punto di riferimento
                    float distSquared =
                        (pointR - data.r[j]) * (pointR - data.r[j]) +
                        (pointG - data.g[j]) * (pointG - data.g[j]) +
                        (pointB - data.b[j]) * (pointB - data.b[j]);

                    // Applica kernel gaussiano per calcolare il peso del contributo
                    if (distSquared < bandwidthSquared) { // Controlla se il punto rientra nella finestra di bandwidth
                        float weight = exp(-distSquared / (2 * bandwidthSquared)); // KERNEL GAUSSIANO
                        localShiftR += weight * data.r[j];
                        localShiftG += weight * data.g[j];
                        localShiftB += weight * data.b[j];
                        localTotalWeight += weight;
                    }
                }

                // Calcola la nuova posizione media pesata del punto (Normalizza)
                if (localTotalWeight > 0) {
                    localShiftR /= localTotalWeight;
                    localShiftG /= localTotalWeight;
                    localShiftB /= localTotalWeight;
                }

                // Calcola la distanza di spostamento (distanza tra vecchia e nuova posizione)
                float shiftDistance =
                    (localShiftR - pointR) * (localShiftR - pointR) +
                    (localShiftG - pointG) * (localShiftG - pointG) +
                    (localShiftB - pointB) * (localShiftB - pointB);

                // Aggiorna il punto con la nuova posizione calcolata
                pointR = localShiftR;
                pointG = localShiftG;
                pointB = localShiftB;

                // Condizione di arresto: se lo spostamento è inferiore alla soglia epsilon, esci dal ciclo (CONDIZIONE CONVERGENZA SODDISFATTA)
                if (shiftDistance < epsilon * epsilon) break;
            } while (true);

            // Salva il risultato nella struttra SoA
            modes.r[i] = pointR;
            modes.g[i] = pointG;
            modes.b[i] = pointB;
        }
    }
}

// Ricostruisce l'immagine, dopo aver applicato il Mean Shift parallelo da SoA a AoS
cv::Mat reconstructFromSoA(const SoAData& modes, int rows, int cols) {
    // Crea una matrice OpenCV di dimensioni rows x cols, inizializzata a zero
    cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC3);

    // Itera su tutti i pixel, utilizzando i dati della struttura SoA
    for (size_t i = 0; i < modes.r.size(); ++i) {
        // Calcola le coordinate (x, y) del pixel a partire dall'indice lineare i
        int x = static_cast<int>(i % cols); // Colonna corrente
        int y = static_cast<int>(i / cols); // Riga "

        // Ricostruisce il colore del pixel utilizzando i dati dei tre canali RGB
        cv::Vec3b color = cv::Vec3b(
            static_cast<uchar>(modes.r[i]),
            static_cast<uchar>(modes.g[i]),
            static_cast<uchar>(modes.b[i])
        );

        // Assegna il colore calcolato al pixel corrispondente nella matrice OpenCV
        result.at<cv::Vec3b>(y, x) = color;
    }
    return result; // Restituisce l'immagine ricostruita
}

// Segmentazione immagine parallela (usa SoA per la segmentazione parallela)
cv::Mat segmentImage_parallel(const cv::Mat& image, float bandwidth, float epsilon) {
    SoAData data = convertToSoA(image); // Converte l'immagine OpenCV da Array of Structures (AoS) a Structure of Arrays (SoA)
    SoAData modes; // Struttura dati per memorizzare i risultati della segmentazione

    // Applica Mean Shift parallelo su SoA
    meanShift_parallel(data, modes, bandwidth, epsilon);

    // Ricostruisci immagine da dati in SoA
    return reconstructFromSoA(modes, image.rows, image.cols);
}
