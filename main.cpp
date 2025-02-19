#include "MeanShiftSeq.h"
#include "MeanShiftParallel.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono> // Necessario per misurare il tempo
#include <omp.h> // Necessario per parallelizzare (OpenMP)

namespace fs = std::filesystem;

int main() {
    // Imposta i parametri per Mean Shift
    float bandwidth = 32.0f; // Ampiezza della finestra di ricerca 32.0f è il migliore (8 non clusterizza (troppi colori), 128 clusterizza troppo (1 solo colore))
    float epsilon = 1.0f;    // Tolleranza per la convergenza

    int resize_factor = 4; // Indica di che fattore scaleremo le immagini. 1:512x512, 2: 256x256, 4:128x128, 8:64x64 //ottimale è 4
    std::string imageDirectory = "./img"; // Specifica la directory delle immagini
    std::string outputDirectory = "./img_results"; // Directory per le immagini segmentate

    // Verifica che la directory esista
    if (!fs::exists(imageDirectory) || !fs::is_directory(imageDirectory)) {
        std::cerr << "Errore: La directory " << imageDirectory << " non esiste o non è una directory!" << std::endl;
        return -1;
    }

    // Scansiona la directory per trovare immagini
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(imageDirectory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    // Verifica se ci sono immagini disponibili
    if (imageFiles.empty()) {
        std::cerr << "Errore: Nessuna immagine trovata nella directory " << imageDirectory << "!" << std::endl;
        return -1;
    }

    // Mostra la lista di immagini all'utente
    std::cout << "Immagini disponibili nella directory " << imageDirectory << ":\n";
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        std::cout << i + 1 << ": " << imageFiles[i] << std::endl;
    }

    // Chiedi all'utente di selezionare un'immagine
    int choice = 0;
    do {
        std::cout << "Scegli un'immagine (1-" << imageFiles.size() << "): ";
        std::cin >> choice;
    } while (choice < 1 || choice > static_cast<int>(imageFiles.size()));

    std::string imagePath = imageFiles[choice - 1];
    std::cout << "Hai scelto: " << imagePath << std::endl;

    // Carica l'immagine
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Errore: Impossibile caricare l'immagine!" << std::endl;
        return -1;
    }

    // Chiedi all'utente quale risoluzione dell'immagine utilizzare
    int resolutionChoice = 0;
    do {
        std::cout << "Scegli la risoluzione dell'immagine:\n";
        std::cout << "1: 512x512\n";
        std::cout << "2: 256x256\n";
        std::cout << "3: 128x128\n";
        std::cout << "4:  64x64\n";
        std::cin >> resolutionChoice;

        if (resolutionChoice < 1 || resolutionChoice > 4) {
            std::cerr << "Scelta non valida. Inserisci un valore tra 1 e 4.\n";
        }
    } while (resolutionChoice < 1 || resolutionChoice > 4);

    // Imposta il resize_factor in base alla scelta dell'utente
    switch (resolutionChoice) {
    case 1: resize_factor = 1; break;  // 512x512
    case 2: resize_factor = 2; break;  // 256x256
    case 3: resize_factor = 4; break;  // 128x128
    case 4: resize_factor = 8; break;  //  64x64
    }

    // Riduci la risoluzione dell'immagine (ad esempio, a metà dimensione)
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(image.cols / resize_factor, image.rows / resize_factor)); // Riduce l'immagine 512x512:Z dove Z è il fattore di ridimensionamento

    // Chiedi all'utente quale versione di MeanShift utilizzare
    int methodChoice = 0;
    do {
        std::cout << "Scegli la versione di MeanShift:\n";
        std::cout << "1: Sequenziale\n";
        std::cout << "2: Parallelo\n";
        std::cin >> methodChoice;

        if (methodChoice != 1 && methodChoice != 2) {
            std::cerr << "Scelta non valida. Inserisci 1 (Sequenziale) o 2 (Parallelo).\n";
        }
    } while (methodChoice != 1 && methodChoice != 2);

    // Se l'utente sceglie la versione parallela, chiedi il numero di thread
    int numThreads = 0;
    if (methodChoice == 2) {
        int maxThreads = omp_get_max_threads(); // Numero massimo di thread disponibili (uso OpenMP)
        std::cout << "Thread disponibili: " << maxThreads << std::endl;

        do {
            std::cout << "Inserisci il numero di thread da utilizzare (1-" << maxThreads << "): ";
            std::cin >> numThreads;
            if (numThreads < 1 || numThreads > maxThreads) {
                std::cerr << "Numero di thread non valido. Inserisci un valore tra 1 e " << maxThreads << ".\n";
            }
        } while (numThreads < 1 || numThreads > maxThreads);

        omp_set_num_threads(numThreads); // Imposta il numero di thread scelto dall'utente (uso OpenMP)
    }

    // Esegui il metodo scelto
    if (methodChoice == 1) {
        // Esegue MeanShift Sequenziale
        auto start_seq = std::chrono::high_resolution_clock::now(); // inizio timer sequenziale
        cv::Mat segmented_seq = segmentImage_seq(resizedImage, bandwidth, epsilon);     // esecuzione segmentazione con MeanShift sequenziale
        auto end_seq = std::chrono::high_resolution_clock::now();   // fine timer sequenziale
        std::chrono::duration<double> elapsed_seq = end_seq - start_seq;                // calcolo tempo sequenziale

        std::cout << "Tempo di esecuzione (sequenziale): " << elapsed_seq.count() << " secondi.\n";

        // Salva il risultato
        std::string outputFileName_segmented_seq =
            outputDirectory + "/sequential_segmented_image_" + std::to_string(choice) + ".jpg"; // Crea il nome del file di output // prende la directory (in modo che stia nella cartella img_results + il 'segemnted' + il numero della scelta)
        cv::imwrite(outputFileName_segmented_seq, segmented_seq);
    } else if (methodChoice == 2) {
        // Esegue MeanShift Parallelo
        auto start_par = std::chrono::high_resolution_clock::now();  // inizio timer parallelo
        cv::Mat segmented_par = segmentImage_parallel(resizedImage, bandwidth, epsilon); // esecuzione segmentazione con MeanShift parallelo
        auto end_par = std::chrono::high_resolution_clock::now();    // fine timer parallelo
        std::chrono::duration<double> elapsed_par = end_par - start_par;                 // calcolo tempo parallelo

        std::cout << "Tempo di esecuzione (parallelo): " << elapsed_par.count()
                  << " secondi. Thread utilizzati: " << numThreads << ".\n";

        // Salva il risultato
        std::string outputFileName_segmented_par =
            outputDirectory + "/parallel_segmented_image_" + std::to_string(choice) + ".jpg"; // Crea il nome del file di output
        cv::imwrite(outputFileName_segmented_par, segmented_par);
    }

    // Crea il nome del file di output
    std::string outputFileName_resized = outputDirectory + "/resized_image_" + std::to_string(choice) + ".jpg"; // prende la directory (in modo che stia nella cartella img_results + il 'segemnted' + il numero della scelta)

    // Salva immagini risultanti nella cartella img_results
    cv::imwrite(outputFileName_resized, resizedImage); // Salva l'immagine ridimensionata

    return 0;
}


