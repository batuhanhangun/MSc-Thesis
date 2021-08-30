/*
This code includes the OpenCV based GPU implementation of the 
"Image enhancement with the application of local and global enhancement methods for dark images" by
Singh et. al https://ieeexplore.ieee.org/document/8071892

Author: Batuhan HANGÜN
*/


#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "libxl.h"


int main()
{
	std::string inputImagePath = "C:\\Users\\batuh\\Desktop\\VS_2017_Projects\\Singh_2017_GPU\\Input_Images\\snow.png";
	cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_UNCHANGED);

	//Deleting all previously written images from the disk and creating new directory
	std::filesystem::remove_all("C:\\Users\\batuh\\Desktop\\VS_2017_Projects\\Singh_2017_GPU\\Results");
	std::filesystem::create_directory("C:\\Users\\batuh\\Desktop\\VS_2017_Projects\\Singh_2017_GPU\\Results");
	
	//Check if image was read successfully
	if (inputImage.empty() == 1) {
		std::cout << "Couldn't read the image: " << inputImagePath << std::endl;
		return EXIT_FAILURE;
	}

	cv::Mat inputImageDummy;



	auto start = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	auto rgbToHsvTimeWarmup = (end - start);
	auto rgbToHsvTime = (end - start);
	auto gaussianBlurTimeWarmup = (end - start);
	auto gaussianBlurTime = (end - start);
	auto subtractTimeWarmup = (end - start);
	auto subtractTime = (end - start);
	auto addTimeWarmup = (end - start);
	auto addTime = (end - start);
	auto localEnhanceTimeWarmup = (end - start);
	auto localEnhanceTime = (end - start);
	auto equalizeHistTimeWarmup = (end - start);
	auto equalizeHistTime = (end - start);
	auto hsvToRgbTimeWarmup = (end - start);
	auto hsvToRgbTime = (end - start);
	auto totalTimeWarmup = (end - start);
	auto totalTime = (end - start);

	int numIter = 100; //50
	int sizeIter = 0;
	int sizeIterLim = 18; //10

	for (sizeIter = 1; sizeIter < sizeIterLim; ++sizeIter) {

		cv::Mat inputImageDummy;

		cv::resize(inputImage, inputImageDummy, cv::Size(), sizeIter, sizeIter);

		cv::cuda::GpuMat inputImageDevice, inputImageHsvDev;
		inputImageDevice.upload(inputImageDummy);


		/* STEP 1 - RGB to HSV Conversion */
		//RGB to HSV conversion
		
		//GPU Warm up
		start = std::chrono::high_resolution_clock::now();
		cv::cuda::cvtColor(inputImageDevice, inputImageHsvDev, cv::COLOR_BGR2HSV);
		end = std::chrono::high_resolution_clock::now();
		rgbToHsvTime = (end - start);

		for (int i = 0; i < numIter; ++i) {
		start = std::chrono::high_resolution_clock::now();
		cv::cuda::cvtColor(inputImageDevice, inputImageHsvDev, cv::COLOR_BGR2HSV);
		end = std::chrono::high_resolution_clock::now();
		rgbToHsvTimeWarmup += (end - start);
		}
		rgbToHsvTimeWarmup /= numIter;


		cv::Mat inputImageHsvHost;
		inputImageHsvDev.download(inputImageHsvHost);


		//Splitting V channel for later use
		cv::cuda::GpuMat inputImageHsvChannelsDev[3];
		cv::cuda::split(inputImageHsvDev, inputImageHsvChannelsDev);

		cv::cuda::GpuMat inputImageHDev = inputImageHsvChannelsDev[0];
		cv::cuda::GpuMat inputImageSDev = inputImageHsvChannelsDev[1];
		cv::cuda::GpuMat inputImageVDev = inputImageHsvChannelsDev[2];

		cv::Mat inputImageHHost, inputImageSHost, inputImageVHost;
		inputImageHDev.download(inputImageHHost);
		inputImageSDev.download(inputImageSHost);
		inputImageVDev.download(inputImageVHost);



		/*--------------------------------------------------*/

	   /* STEP 2 - LOCAL ENHANCEMENT */
	   // 1.Blurring of the image
	   // 2.Subtracting the blurred image from the original image to make the mask
	   // 3.Adding mask to the original image(Sharpening the image)

		//1.Blurring of the image
		cv::cuda::GpuMat blurredImageDev = inputImageVDev.clone();
		cv::Ptr<cv::cuda::Filter> filter5x5 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1);

		//GPU Warm up
		start = std::chrono::high_resolution_clock::now();
		filter5x5->apply(inputImageVDev, blurredImageDev);
		end = std::chrono::high_resolution_clock::now();
		gaussianBlurTime = (end - start);

		cv::Mat blurredImageHost;
		blurredImageDev.download(blurredImageHost);
		//cv::imshow("Blurred Image", blurredImageHost);
		//cv::waitKey();


		for (int i = 0; i < numIter; ++i) {
		start = std::chrono::high_resolution_clock::now();
		filter5x5->apply(inputImageVDev, blurredImageDev);
		end = std::chrono::high_resolution_clock::now();
		gaussianBlurTimeWarmup += (end - start);
		}
		gaussianBlurTimeWarmup /= numIter;



		//2.Subtracting the blurred image from the original image to obtain the image mask
		cv::cuda::GpuMat imageMaskDev = inputImageVDev.clone();

		//GPU Warm up
		start = std::chrono::high_resolution_clock::now();
		cv::cuda::subtract(inputImageVDev, blurredImageDev, imageMaskDev);
		end = std::chrono::high_resolution_clock::now();
		subtractTime = (end - start);

		for (int i = 0; i < numIter; ++i) {
			start = std::chrono::high_resolution_clock::now();
			cv::cuda::subtract(inputImageVDev, blurredImageDev, imageMaskDev);
			end = std::chrono::high_resolution_clock::now();
			subtractTimeWarmup += (end - start);
		}
		subtractTimeWarmup /= numIter;

		cv::Mat imageMaskHost;
		imageMaskDev.download(imageMaskHost);


		//3.Adding mask to the original image(Sharpening the image)
		cv::cuda::GpuMat sharpenedImageDev;
		int weightVal = 10;

		//GPU Warm up
		start = std::chrono::high_resolution_clock::now();
		cv::cuda::scaleAdd(imageMaskDev, weightVal, inputImageVDev, sharpenedImageDev);
		end = std::chrono::high_resolution_clock::now();
		addTime = (end - start);

		for (int i = 0; i < numIter; ++i) {
			start = std::chrono::high_resolution_clock::now();
			cv::cuda::scaleAdd(imageMaskDev, weightVal, inputImageVDev, sharpenedImageDev);
			end = std::chrono::high_resolution_clock::now();
			addTimeWarmup += (end - start);
		}
		addTimeWarmup /= numIter;

		localEnhanceTime = gaussianBlurTime + subtractTime + addTime;
		localEnhanceTimeWarmup = gaussianBlurTimeWarmup + subtractTimeWarmup + addTimeWarmup;


		cv::Mat sharpenedImageHost;
		sharpenedImageDev.download(sharpenedImageHost);


		/* STEP 3 - GLOBAL ENHANCEMENT*/
		// Histogram equalization
		cv::cuda::GpuMat globallyEnhancedImageDev;

		start = std::chrono::high_resolution_clock::now();

		//GPU Warm up
		cv::cuda::equalizeHist(sharpenedImageDev, globallyEnhancedImageDev);
		end = std::chrono::high_resolution_clock::now();
		equalizeHistTime = (end - start);

		for (int i = 0; i < numIter; ++i) {
			start = std::chrono::high_resolution_clock::now();
			cv::cuda::equalizeHist(sharpenedImageDev, globallyEnhancedImageDev);
			end = std::chrono::high_resolution_clock::now();
			equalizeHistTimeWarmup += (end - start);
		}
		equalizeHistTimeWarmup /= numIter;

		cv::Mat globallyEnhancedImageHost;
		globallyEnhancedImageDev.download(globallyEnhancedImageHost);


		/* Merging of enhanced V band and H, S bands */
		cv::cuda::GpuMat outputImageDev;
		const int vecSize = 3;
		cv::cuda::GpuMat channelsVec[vecSize] = { inputImageHDev, inputImageSDev, globallyEnhancedImageDev };
		cv::cuda::merge(channelsVec, vecSize, outputImageDev);
		cv::cuda::GpuMat dummyImageDev = outputImageDev.clone();


		//HSV to RGB conversion
		start = std::chrono::high_resolution_clock::now();

		//GPU Warm up
		cv::cuda::cvtColor(dummyImageDev, outputImageDev, cv::COLOR_HSV2BGR);
		end = std::chrono::high_resolution_clock::now();
		hsvToRgbTime = (end - start);


		for (int i = 0; i < numIter; ++i) {
			start = std::chrono::high_resolution_clock::now();
			cv::cuda::cvtColor(dummyImageDev, outputImageDev, cv::COLOR_HSV2BGR);
			end = std::chrono::high_resolution_clock::now();
			hsvToRgbTimeWarmup += (end - start);
		}
		hsvToRgbTimeWarmup /= numIter;

		totalTime = rgbToHsvTime + localEnhanceTime + equalizeHistTime + hsvToRgbTime;
		totalTimeWarmup = rgbToHsvTimeWarmup + localEnhanceTimeWarmup + equalizeHistTimeWarmup + hsvToRgbTimeWarmup;




		cv::Mat outputImageHost;
		outputImageDev.download(outputImageHost);

		//Writing results
		cv::imwrite("C:\\Users\\batuh\\Desktop\\VS_2017_Projects\\Singh_2017_GPU\\Results\\results.png", outputImageHost);

		libxl::Book* book = xlCreateBook(); // xlCreateXMLBook() for xlsx

		if (book)
		{
			libxl::Sheet* sheet;
			std::wstring rowSizeStr = std::to_wstring(inputImageHsvHost.rows);
			std::wstring colSizeStr = std::to_wstring(inputImageHsvHost.cols);
			std::wstring sizeString = rowSizeStr + L"X" + colSizeStr;
			const wchar_t* sizeWchar = sizeString.c_str();
			std::wstring fileNameDest = L"C:\\Users\\batuh\\Desktop\\TimeEvalResults\\Singh2017_GPU\\Singh2017_GPU_";
			std::wstring fileNameString = fileNameDest + sizeWchar + L".xls";
			const wchar_t* fileNameWchar = fileNameString.c_str();


			sheet = book->addSheet(L"Warm Up");
			if (sheet)
			{
				sheet->writeStr(1, 0, L"Image Size");
				sheet->writeNum(2, 0, inputImageHsvHost.rows * inputImageHsvHost.cols);
				sheet->writeStr(3, 0, sizeWchar);
				/////////////////////////////////////////
				sheet->writeStr(1, 1, L"rgbToHsvTime");
				sheet->writeNum(2, 1, std::chrono::duration <double, std::milli>(rgbToHsvTimeWarmup).count());
				/////////////////////////////////////////
				sheet->writeStr(1, 2, L"localEnhanceTime");
				sheet->writeNum(2, 2, std::chrono::duration <double, std::milli>(localEnhanceTimeWarmup).count());
				/////////////////////////////////////////
				sheet->writeStr(1, 3, L"globalEnhanceTime");
				sheet->writeNum(2, 3, std::chrono::duration <double, std::milli>(equalizeHistTimeWarmup).count());
				/////////////////////////////////////////
				sheet->writeStr(1, 4, L"hsvToRgbTime");
				sheet->writeNum(2, 4, std::chrono::duration <double, std::milli>(hsvToRgbTimeWarmup).count());
				/////////////////////////////////////////
				sheet->writeStr(1, 5, L"totalTime");
				sheet->writeNum(2, 5, std::chrono::duration <double, std::milli>(totalTimeWarmup).count());
			}
			sheet = book->addSheet(L"No Warm Up");
			if (sheet)
			{
				sheet->writeStr(1, 0, L"Image Size");
				sheet->writeNum(2, 0, inputImageHsvHost.rows * inputImageHsvHost.cols);
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

	return 0;
}


