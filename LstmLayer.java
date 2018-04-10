package com.albert.lstm;

import java.util.ArrayList;

class LstmLayer {
	/*
	 * #权重与偏置的角标 C都代表临时单元状态，其他情况临时单元状态要写成tempC
	 * */
	
	//输入向量的维度,单元状态向量的维度
	public int inputWidth;
	public int stateWidth;
	
	//学习速率
	public double learningRate;
	
	//权重矩阵与偏置矩阵,所有都将被随机初始化
	//遗忘门
	public Mat weigthFH;
	public Mat weigthFX;
	public Mat biasF;
	//输入门
	public Mat weigthIH;
	public Mat weigthIX;
	public Mat biasI;
	//输出门
	public Mat weigthOH;
	public Mat weigthOX;
	public Mat biasO;
	//临时单元状态
	public Mat weigthCH;
	public Mat weigthCX;
	public Mat biasC;
	
	//当前时刻(步骤)初始化为0
	private int time = 0;
	
	//用一个ArrayList装载各个时刻的输出向量h 和 单元状态c
	public ArrayList<Mat> everytimeStateOfH = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfC = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfF = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfI = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfO = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfTempC = new ArrayList<Mat>();

	// ---------
	//用一个ArrayList装载各个时刻的输出状态h ,输出门o ,输入门i ,遗忘门f ,临时单元状态tempC 的误差
	public ArrayList<Mat> everytimeDeltaOfH = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfO = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfI = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfF = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfTempC = new ArrayList<Mat>();
	
	//三门和临时单元单元状态的权重与偏置的总梯度值(总梯度值!!!)
	public Mat totalGradOfWfh;
	public Mat totalGradOfWih;
	public Mat totalGradOfWoh;
	public Mat totalGradOfWch;
	
	public Mat gradOfWfx;
	public Mat gradOfWix;
	public Mat gradOfWox;
	public Mat gradOfWcx;
	
	public Mat totalGradOfBf;
	public Mat totalGradOfBi;
	public Mat totalGradOfBo;
	public Mat totalGradOfBc;
	
	
	/*
	 * ============================================================================
	 * 								变量/方法 -分割线
	 * ============================================================================
	 * */
	
	/*
	 * 初始化权重与偏置矩阵
	 * 
	 * */
	public void initWeight() {
		double[][] wfhArray = new double[stateWidth][stateWidth];
		double[][] wfxArray = new double[stateWidth][inputWidth];
		double[][] bfArray = new double[stateWidth][1];
		
		double[][] wihArray = new double[stateWidth][stateWidth];
		double[][] wixArray = new double[stateWidth][inputWidth];
		double[][] biArray = new double[stateWidth][1];
		
		double[][] wohArray = new double[stateWidth][stateWidth];
		double[][] woxArray = new double[stateWidth][inputWidth];
		double[][] boArray = new double[stateWidth][1];
		
		double[][] wchArray = new double[stateWidth][stateWidth];
		double[][] wcxArray = new double[stateWidth][inputWidth];
		double[][] bcArray = new double[stateWidth][1];
		
		for (int i = 0; i < stateWidth; i++) {
			bfArray[i][0] = (Math.random() - 0.5) * 2 * Math.exp(-4);
			biArray[i][0] = (Math.random() - 0.5) * 2 * Math.exp(-4);
			boArray[i][0] = (Math.random() - 0.5) * 2 * Math.exp(-4);
			bcArray[i][0] = (Math.random() - 0.5) * 2 * Math.exp(-4);
			
			for (int j = 0; j < stateWidth; j++) {
				wfhArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);				
				wihArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);			
				wohArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);				
				wchArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);
			}	
			for (int j = 0; j < inputWidth; j++) {
				wfxArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);			
				wixArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);				
				woxArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);				
				wcxArray[i][j] = (Math.random() - 0.5) * 2 * Math.exp(-4);
			}
			
		}
		this.weigthFH = new Mat(wfhArray);
		this.weigthFX = new Mat(wfxArray);
		this.biasF = new Mat(bfArray);
		
		this.weigthIH = new Mat(wihArray);
		this.weigthIX = new Mat(wixArray);
		this.biasI = new Mat(biArray);
		
		this.weigthOH = new Mat(wohArray);
		this.weigthOX = new Mat(woxArray);
		this.biasO = new Mat(boArray);
		
		this.weigthCH = new Mat(wchArray);
		this.weigthCX = new Mat(wcxArray);
		this.biasC = new Mat(bcArray);
		
	}
	
	/*
	 * 初始化前馈计算中的各个门以及单元状态与临时单元状态
	 * everytimeStateOfH/everytimeStateOfC/everytimeStateOfTempC/everytimeStateOfF/everytimeStateOfI/everytimeStateOfO
	 * */
	public void initForwardState() {
		everytimeStateOfH.clear();
		everytimeStateOfH.add(Mat.zeroMat(stateWidth, 1));
		
		everytimeStateOfC.clear();
		everytimeStateOfC.add(Mat.zeroMat(stateWidth, 1));
		
		everytimeStateOfTempC.clear();
		everytimeStateOfTempC.add(Mat.zeroMat(stateWidth, 1));
		
		everytimeStateOfF.clear();
		everytimeStateOfF.add(Mat.zeroMat(stateWidth, 1));
		
		everytimeStateOfI.clear();
		everytimeStateOfI.add(Mat.zeroMat(stateWidth, 1));
		
		everytimeStateOfO.clear();
		everytimeStateOfO.add(Mat.zeroMat(stateWidth, 1));
	}

	/*
	 * 构造方法
	 * 各种初始化代码都在里面
	 * */
	public LstmLayer(int inputWidth, int stateWidth, double learningRate) {
		this.inputWidth = inputWidth;
		this.stateWidth = stateWidth;
		this.learningRate = learningRate;
		initWeight();//初始化权重与偏置
				
		//将该时刻(第零时刻)的(前馈计算值)输出向量h 和单元状态c,临时单元状态,遗忘门，输入门，输出门装载入
		initForwardState();
		
		//初始化权重与配置的梯度
		totalGradOfWfh = Mat.zeroMat(stateWidth, stateWidth);
		totalGradOfWoh = Mat.zeroMat(stateWidth, stateWidth);
		totalGradOfWih = Mat.zeroMat(stateWidth, stateWidth);
		totalGradOfWch = Mat.zeroMat(stateWidth, stateWidth);
		
		gradOfWfx = Mat.zeroMat(stateWidth, inputWidth);
		gradOfWox = Mat.zeroMat(stateWidth, inputWidth);
		gradOfWix = Mat.zeroMat(stateWidth, inputWidth);
		gradOfWcx = Mat.zeroMat(stateWidth, inputWidth);

		totalGradOfBf = Mat.zeroMat(stateWidth, 1);
		totalGradOfBi = Mat.zeroMat(stateWidth, 1);
		totalGradOfBo = Mat.zeroMat(stateWidth, 1);
		totalGradOfBc = Mat.zeroMat(stateWidth, 1);

	}
	
	/*
	 * 前馈计算
	 * stateOfX : 当前时刻的输入向量
	 * 计算各时刻的前馈值，并装载入ArrayList
	 * */
	public void forward(Mat stateOfX) {
		this.time ++;
		
		//遗忘门计算
		Mat fGate = calculateGate(stateOfX, weigthFH, weigthFX, biasF, true);
		everytimeStateOfF.add(fGate);
		//输入门计算
		Mat iGate = calculateGate(stateOfX, weigthIH, weigthIX, biasI, true);
		everytimeStateOfI.add(iGate);
		//输出门计算
		Mat oGate = calculateGate(stateOfX, weigthOH, weigthOX, biasO, true);
		everytimeStateOfO.add(oGate);
		//临时单元状态
		Mat tempC = calculateGate(stateOfX, weigthCH, weigthCX, biasC, false);
		everytimeStateOfTempC.add(tempC);
		//单元状态 C = fGate .* lastC + iGate .* tempC 
		// lastC = everytimeStateOfC(time - 1);
		Mat C = Mat.plus(Mat.mulByEle(fGate, everytimeStateOfC.get(this.time - 1)), Mat.mulByEle(iGate, tempC));
		//此时刻的单元状态装入
		everytimeStateOfC.add(C);
		
		//输出向量H = oGate .* C
		Mat H = Mat.mulByEle(oGate, C);
		//此时刻的输出向量装入
		everytimeStateOfH.add(H);
		
	}
	
	/*
	 * 通用的门计算方法
	 * stateOfX : 当前时刻的输入向量
	 * flag : 标志位 true ：三个门 
	 * 			    false：临时单元状态
	 * */
	public Mat calculateGate(Mat stateOfX, Mat weigthH, Mat weigthX, Mat bias, boolean flag) {
		//上一时刻的输出
		Mat lastH = everytimeStateOfH.get(time - 1);
		
		//此时刻的加权输入 net = weigthH * lastH + weigthX * stateOfX + bias
		Mat net = Mat.plus(Mat.plus(Mat.mul(weigthH, lastH), Mat.mul(weigthX, stateOfX)), bias);
		
		//门矩阵
		Mat gate = flag ? Activator.sigmoid(net) : Activator.tanh(net);
		return gate;
	}

	/*
	 * 后向计算
	 * */
	public void backward(Mat deltaH, Mat stateOfX) {
		//计算误差
		calculateDelta(deltaH);
		
		//计算梯度
		calculateGradient(stateOfX);
		
	}
	
	/*
	 * 计算误差
	 * */
	public void calculateDelta(Mat deltaH) {
		//初始化各个时刻的输出状态h ,输出门o ,输入门i ,遗忘门f ,临时单元状态tempC 的误差
		//Mat init = Mat.zeroMat(stateWidth, 1);
		for (int i = 0; i < this.time + 1; i++) {		
			everytimeDeltaOfH.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfO.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfI.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfF.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfTempC.add(Mat.zeroMat(stateWidth, 1));
		}

		//保存当前时刻的输出误差
		everytimeDeltaOfH.set(everytimeDeltaOfH.size() - 1, deltaH);
		
		//计算各个时刻的各个误差项
		for (int i = this.time; i > 0; i--) {
			//第i时刻的前馈计算值
			Mat fGate_i = everytimeStateOfF.get(i);
			Mat iGate_i = everytimeStateOfI.get(i);
			Mat oGate_i = everytimeStateOfO.get(i);
			Mat tempC_i = everytimeStateOfTempC.get(i);
			Mat C_i = everytimeStateOfC.get(i);
			Mat lastC = everytimeStateOfC.get(i - 1);
			//计算tanh(C)
			Mat tanhC_i = Activator.tanh(C_i);
			//此时刻(第i时刻)的deltaH_i值
			Mat deltaH_i = everytimeDeltaOfH.get(i);
			
			Mat deltaO_i = Mat.mulByEle(deltaH_i, tanhC_i, Activator.sigmoidDerivative(oGate_i));//计算deltaO
			Mat deltaF_i = Mat.mulByEle(deltaH_i, oGate_i, Activator.tanhDerivative(C_i), lastC, Activator.sigmoidDerivative(fGate_i));//计算deltaO
			Mat deltaI_i = Mat.mulByEle(deltaH_i, oGate_i, Activator.tanhDerivative(C_i), tempC_i, Activator.sigmoidDerivative(iGate_i));//计算deltaO
			Mat deltaTempC_i = Mat.mulByEle(deltaH_i, oGate_i, Activator.tanhDerivative(C_i), iGate_i, Activator.tanhDerivative(tempC_i));
			
			//计算第 i-1 时刻 的deltaH值lastDeltaH
			Mat lastDeltaH = Mat.transpose(
										Mat.plus(
												Mat.mul(Mat.transpose(deltaO_i), this.weigthOH),
												Mat.mul(Mat.transpose(deltaF_i), this.weigthFH),
												Mat.mul(Mat.transpose(deltaI_i), this.weigthIH),
												Mat.mul(Mat.transpose(deltaTempC_i), this.weigthCH)
												));
					
			//保存delta误差值
			everytimeDeltaOfH.set(i - 1, lastDeltaH);
			everytimeDeltaOfF.set(i, deltaF_i);
			everytimeDeltaOfI.set(i, deltaI_i);
			everytimeDeltaOfO.set(i, deltaO_i);
			everytimeDeltaOfTempC.set(i, deltaTempC_i);	
		}		
	}
	
	/*
	 * 计算梯度
	 * */
	public void calculateGradient(Mat stateOfX) {
		
		for (int i = this.time; i > 0; i--) {
			Mat lastH = everytimeStateOfH.get(i - 1);
			//计算当前时刻i的W*h和B*梯度
			//gradOfWfh_i = deltaF_i * H_i-1
			//gradOfBf = deltaF_i;
			Mat gradOfWfh_i = Mat.mul(everytimeDeltaOfF.get(i), Mat.transpose(lastH));
			Mat gradOfWih_i = Mat.mul(everytimeDeltaOfI.get(i), Mat.transpose(lastH));
			Mat gradOfWoh_i = Mat.mul(everytimeDeltaOfO.get(i), Mat.transpose(lastH));
			Mat gradOfWch_i = Mat.mul(everytimeDeltaOfTempC.get(i), Mat.transpose(lastH));
			Mat gradOfBf = everytimeDeltaOfF.get(i);
			Mat gradOfBi = everytimeDeltaOfI.get(i);
			Mat gradOfBo = everytimeDeltaOfO.get(i);
			Mat gradOfBc = everytimeDeltaOfTempC.get(i);
			
			//总的梯度是各个时刻梯度之和
			totalGradOfWfh = Mat.plus(totalGradOfWfh, gradOfWfh_i);
			totalGradOfWih = Mat.plus(totalGradOfWih, gradOfWih_i);
			totalGradOfWoh = Mat.plus(totalGradOfWoh, gradOfWoh_i);
			totalGradOfWch = Mat.plus(totalGradOfWch, gradOfWch_i);
			
			totalGradOfBf = Mat.plus(totalGradOfBf, gradOfBf);
			totalGradOfBi = Mat.plus(totalGradOfBi, gradOfBi);
			totalGradOfBo = Mat.plus(totalGradOfBo, gradOfBo);
			totalGradOfBc = Mat.plus(totalGradOfBc, gradOfBc);
				
		}
		//计算当前时刻的W*x的权重
		gradOfWfx = Mat.mul(everytimeDeltaOfF.get(everytimeDeltaOfF.size() - 1), Mat.transpose(stateOfX));//Mat.transpose(stateOfX)		
		gradOfWix = Mat.mul(everytimeDeltaOfI.get(everytimeDeltaOfI.size() - 1), Mat.transpose(stateOfX));
		gradOfWox = Mat.mul(everytimeDeltaOfO.get(everytimeDeltaOfO.size() - 1), Mat.transpose(stateOfX));
		gradOfWcx = Mat.mul(everytimeDeltaOfTempC.get(everytimeDeltaOfTempC.size() - 1), Mat.transpose(stateOfX));
	}
	
	/*
	 * 用所求的总梯度更新权重
	 * 
	 * */
	public void updateWeigth() {
		weigthFH = Mat.subs(weigthFH, Mat.scalMul(learningRate, totalGradOfWfh));
		weigthFX = Mat.subs(weigthFX, Mat.scalMul(learningRate, gradOfWfx));
		biasF = Mat.subs(biasF, Mat.scalMul(learningRate, totalGradOfBf));
		
		weigthIH = Mat.subs(weigthIH, Mat.scalMul(learningRate, totalGradOfWih));
		weigthIX = Mat.subs(weigthIX, Mat.scalMul(learningRate, gradOfWix));
		biasI = Mat.subs(biasI, Mat.scalMul(learningRate, totalGradOfBi));
		
		weigthOH = Mat.subs(weigthOH, Mat.scalMul(learningRate, totalGradOfWoh));
		weigthOX = Mat.subs(weigthOX, Mat.scalMul(learningRate, gradOfWox));
		biasO = Mat.subs(biasO, Mat.scalMul(learningRate, totalGradOfBo));
		
		weigthCH = Mat.subs(weigthCH, Mat.scalMul(learningRate, totalGradOfWch));
		weigthCX = Mat.subs(weigthCX, Mat.scalMul(learningRate, gradOfWcx));
		biasC = Mat.subs(biasC, Mat.scalMul(learningRate, totalGradOfBc));
	}
	
	/*
	 * 调试：梯度检验中要用的状态重置
	 * 状态重置
	 * */
	public void resetState() {
		this.time = 0;
		initForwardState();		
	}
		
	/*
	 * 数据集的训练
	 * */
	public void train() {
		
	}
	
	
	public static void main(String[] args) {
		
	}
}



























