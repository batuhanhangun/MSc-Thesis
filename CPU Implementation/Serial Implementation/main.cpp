/*
OpenCV (C++) Implemantation of "Image enhancement with the application of local and global enhancement methods for dark images" by
Singh et. al https://ieeexplore.ieee.org/document/8071892
This implementation only includes CPU version
Author: Batuhan HANGÜN
*/

#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <chrono>
#include "libxl.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main(int argc, char *argv[])
{


	//Reading the input image
	std::string inputImagePath = "C:\\Users\\batuh\\Desktop\\VS_2017_Projects\\Singh_2017\\Input_Images\\snow.png";

	cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_UNCHANGED);
	//Check if image was read successfully
	if (inputImage.empty() == 1) {
		std::cout << "Couldn't read the image: " << inputImagePath << std::endl;
		return EXIT_FAILURE;
	}
	cv::Mat inputImageDummy;

	
	std::cout << "Default number of threads of OpenCV:  " << cv::getNumThreads() << std::endl;

	int numThreads = 0;
	int threadLim = 5; 
	int numIter = 100; 
	int sizeIter = 0;
	int sizeIterLim = 18; 

	for (numThreads = 0; numThreads < threadLim; ++numThreads) {

		cv::setNumThreads(pow(2,numThreads));
		std::cout << "Current number of threads were set to: " << cv::getNumThreads() << std::endl;

		for (sizeIter = 1; sizeIter < sizeIterLim; ++sizeIter) {

			cv::resize(inputImage, inputImageDummy, cv::Size(), sizeIter, sizeIter);


			//Creating time evaluation variables
			auto start = std::chrono::high_resolution_clock::now();
			auto end = std::chrono::high_resolution_clock::now();
			auto rgbToHsvTime = (end - start);
			auto gaussianBlurTime = (end - start);
			auto subtractTime = (end - start);
			auto addTime = (end - start);
			auto localEnhanceTime = (end - start);
			auto equalizeHistTime = (end - start);
			auto hsvToRgbTime = (end - start);
			auto totalTime = (end - start);

			//RGB to HSV conversion
			cv::Mat inputImageHsv;

			for (int i = 0; i < numIter; ++i) {
				start = std::chrono::high_resolution_clock::now();
				cv::cvtColor(inputImageDummy, inputImageHsv, cv::COLOR_BGR2HSV); // Convert the image to HSV
				end = std::chrono::high_resolution_clock::now();
				rgbToHsvTime += (end - start);
			}
			rgbToHsvTime /= numIter;



			//Splitting V channel for later use
			cv::Mat inputImageHsvChannels[3];
			cv::split(inputImageHsv, inputImageHsvChannels);
			cv::Mat inputImageH = inputImageHsvChannels[0];
			cv::Mat inputImageS = inputImageHsvChannels[1];
			cv::Mat inputImageV = inputImageHsvChannels[2];

			/*--------------------------------------------------*/


		   /* STEP 2 - LOCAL ENHANCEMENT */
		   // 1.Blurring of the image
		   // 2.Subtracting the blurred image from the original image to make the mask
		   // 3.Adding mask to the original image(Sharpening the image)

			cv::Mat blurredImage = inputImageV.clone();
			//1.Blurring of the image
			for (int i = 0; i < numIter; ++i) {
				start = std::chrono::high_resolution_clock::now();
				cv::GaussianBlur(inputImageV, blurredImage, cv::Size(5, 5), 0, 0);
				end = std::chrono::high_resolution_clock::now();
				gaussianBlurTime += (end - start);
			}
			gaussianBlurTime /= numIter;


			cv::Mat imageMask = inputImageV.clone();

			//2.Subtracting the blurred image from the original image to make the mask
			for (int i = 0; i < numIter; ++i) {
				start = std::chrono::high_resolution_clock::now();
				cv::subtract(inputImageV, blurredImage, imageMask);
				end = std::chrono::high_resolution_clock::now();
				subtractTime += (end - start);
			}
			subtractTime /= numIter;


			//3.Adding mask to the original image(Sharpening the image)
			int weight = 10;
			cv::Mat sharpenedImage;

			for (int i = 0; i < numIter; ++i) {
				start = std::chrono::high_resolution_clock::now();
				cv::add(inputImageV, (imageMask * weight), sharpenedImage);
				end = std::chrono::high_resolution_clock::now();
				addTime += (end - start);
			}
			addTime /= numIter;


			/* STEP 3 - GLOBAL ENHANCEMENT*/
			// Histogram equalization
			cv::Mat globallyEnhancedImage;

			for (int i = 0; i < numIter; ++i) {
				start = std::chrono::high_resolution_clock::now();
				cv::equalizeHist(sharpenedImage, globallyEnhancedImage);
				end = std::chrono::high_resolution_clock::now();
				equalizeHistTime += (end - start);
			}
			equalizeHistTime /= numIter;

			/* Merging of enhanced H band and S, V bands */
			cv::Mat outputImage;
			cv::Mat channels[3] = { inputImageH, inputImageS, globallyEnhancedImage };
			cv::merge(channels, 3, outputImage);
			cv::Mat outputImageTemp = outputImage.clone(); //Dummy image object for time evaluation

			//HSV to RGB conversion
			for (int i = 0; i < numIter; ++i) {
				start = std::chrono::high_resolution_clock::now();
				cv::cvtColor(outputImage, outputImage, cv::COLOR_HSV2BGR); // Convert the image to HSV
				end = std::chrono::high_resolution_clock::now();
				hsvToRgbTime += (end - start);
			}
			hsvToRgbTime /= numIter;
			
			//Writing results
			cv::imwrite("C:\\Users\\batuh\\Desktop\\VS_2017_Projects\\Singh_2017\\Results\\results.png", outputImage);
			localEnhanceTime = gaussianBlurTime + subtractTime + addTime;
			totalTime = rgbToHsvTime + localEnhanceTime + equalizeHistTime + hsvToRgbTime;

			libxl::Book* book = xlCreateBook(); // xlCreateXMLBook() for xlsx

			if (book)
			{
				libxl::Sheet* sheet;
				std::wstring rowSizeStr = std::to_wstring(inputImageHsv.rows);
				std::wstring colSizeStr = std::to_wstring(inputImageHsv.cols);
				std::wstring sizeString = rowSizeStr + L"X" + colSizeStr;
				const wchar_t* sizeWchar = sizeString.c_str();
				std::wstring threadNumStr = std::to_wstring(static_cast<int>(pow(2, numThreads)));
				std::wstring fileNameDest = L"C:\\Users\\batuh\\Desktop\\TimeEvalResults\\Singh2017\\Singh2017_numthreads_" + threadNumStr + L"\\Singh2017_numthreads_" + threadNumStr + L"_";

				std::wstring fileNameString = fileNameDest + sizeWchar + L".xls";
				const wchar_t* fileNameWchar = fileNameString.c_str();
				std::filesystem::create_directories(L"C:\\Users\\batuh\\Desktop\\TimeEvalResults\\Singh2017\\Singh2017_numthreads_" + threadNumStr);


				int sheetVal = sizeIter; 

				sheet = book->addSheet(L"Sheet1");
				if (sheet)
				{
					sheet->writeStr(1, 0, L"Image Size");
					sheet->writeNum(2, 0, inputImageHsv.rows * inputImageHsv.cols);
					sheet->writeStr(3, 0, sizeWchar);
					/////////////////////////////////////////
					sheet->writeStr(1, 1, L"rgbToHsvTime");
					sheet->writeNum(2, 1, std::chrono::duration <double, std::milli>(rgbToHsvTime).count());
					/////////////////////////////////////////
					sheet->writeStr(1, 2, L"localEnhanceTime");
					sheet->writeNum(2, 2, std::chrono::duration <double, std::milli>(localEnhanceTime).count());
					/////////////////////////////////////////
					sheet->writeStr(1, 3, L"globalEnhanceTime");
					sheet->writeNum(2, 3, std::chrono::duration <double, std::milli>(equalizeHistTime).count());
					/////////////////////////////////////////
					sheet->writeStr(1, 4, L"hsvToRgbTime");
					sheet->writeNum(2, 4, std::chrono::duration <double, std::milli>(hsvToRgbTime).count());
					/////////////////////////////////////////
					sheet->writeStr(1, 5, L"totalTime");
					sheet->writeNum(2, 5, std::chrono::duration <double, std::milli>(totalTime).count());
				}
				book->save(fileNameWchar);
				book->release();

			}
		}

	}
  return 0;
}



