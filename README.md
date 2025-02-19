# 🎨 **Mean Shift Image Segmentation**
🚀 *Comparison between **sequential** and **parallel** implementations*

> This project explores the **Mean Shift** clustering for image segmentation, comparing a **sequential** version with a **parallel** one optimized with **OpenMP**.

---  

## 🛠️ **Features**
- ✂️ **Image segmentation** using **Mean Shift**.
- ⏱️ **Performance comparison** between sequential and parallel versions.
- 📁 Support for `.jpg`, `.png`, and `.bmp` image files.
- 💾 Saves results in a dedicated folder.

---  

## 🖥️ **Languages and Libraries Used**
✅ **C++**  
✅ **OpenCV** (for image processing).  
✅ **OpenMP** (for parallelization).  
✅ **MinGW** (for compilation).  
✅ **CMake** (for cross-platform builds).

---  

## ⚙️ **Usage**
1. **Add images** to the `./img` directory. Some sample 512x512 images are already included in the project.
2. **Choose the version to use** and **run the program**.
3. **Select an image** from the list displayed in the terminal (images in the `./img` directory) and **choose which Mean Shift version** to use.
4. Check the results in the `./img_results` directory and compare the execution time of the chosen version.

💡 *The program will display execution times for the selected implementation!*

---  

## 🔧 **Customizable Parameters**
- 📏 **Bandwidth**: The search window size (*larger = less detailed segmentation*).
- 🎯 **Epsilon**: Convergence threshold (*smaller = more precise results, but slower*).
- 🔍 **Resize Factor**: Rescales the image for faster testing (1 = no scaling).

---  

## 🗂️ **Resulting Images**
Images will be saved in:
- **`./img_results/sequential_segmented_image_X.jpg`**
- **`./img_results/parallel_segmented_image_X.jpg`**
- **`./img_results/resized_image_X.jpg`**

---  

## 📄 **Report**
A copy of the report (in Italian) can be found [here📄](./report/Parallel_Computing_First_Course_Project_Giovanni_Stefanini.pdf).

A copy of the presentation can be found [here📄](./report/Presentation_Project_1_PC_Giovanni_Stefanini.pdf).

---  

## 🎉 **Contributions**
💡 This project was developed by **Giovanni Stefanini** as part of the Parallel Computing course.

---  

### 👀 **Visual Demo**
Mean Shift segmentation applied to the sample images included in the project:

| **Original Input** | **Sequential Segmentation** | **Parallel Segmentation** |  
|---------------------|------------------------------|------------------------------|  
| ![Original](./cmake-build-debug/img_results/resized_image_1.jpg) | ![Sequential](./cmake-build-debug/img_results/sequential_segmented_image_1.jpg) | ![Parallel](./cmake-build-debug/img_results/parallel_segmented_image_1.jpg) |  
| ![Original](./cmake-build-debug/img_results/resized_image_2.jpg) | ![Sequential](./cmake-build-debug/img_results/sequential_segmented_image_2.jpg) | ![Parallel](./cmake-build-debug/img_results/parallel_segmented_image_2.jpg) |  
| ![Original](./cmake-build-debug/img_results/resized_image_3.jpg) | ![Sequential](./cmake-build-debug/img_results/sequential_segmented_image_3.jpg) | ![Parallel](./cmake-build-debug/img_results/parallel_segmented_image_3.jpg) |  

