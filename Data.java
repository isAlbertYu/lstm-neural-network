package com.albert.lstm;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;

class Data {
	public String allString = "";
	public Mat allData;
	
	public Data(String filename) {
		File file = new File(filename);
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

			String temp = "";

			int count = 0;
			
			while ((temp = br.readLine()) != null) {
				allString = allString + temp + " ";			
				count ++;
			}
			String[] strArr = allString.split("\\s");
			double[][] dat = new double[count][1];
			for (int i = 0; i < count; i++) {
				dat[i][0] = Double.parseDouble(strArr[i]);		
			}
			
			allData = new Mat(dat);
			
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
		
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Data d = new Data("D:\\课程学习资料\\我的编程研究\\黄金价格预测\\goldPrice.txt");

		Mat.printMat(d.allData);
	}

}
