package com.albert.lstm;

import java.util.ArrayList;

class LstmLayer {
	/*
	 * #Ȩ����ƫ�õĽǱ� C��������ʱ��Ԫ״̬�����������ʱ��Ԫ״̬Ҫд��tempC
	 * */
	
	//����������ά��,��Ԫ״̬������ά��
	public int inputWidth;
	public int stateWidth;
	
	//ѧϰ����
	public double learningRate;
	
	//Ȩ�ؾ�����ƫ�þ���,���ж����������ʼ��
	//������
	public Mat weigthFH;
	public Mat weigthFX;
	public Mat biasF;
	//������
	public Mat weigthIH;
	public Mat weigthIX;
	public Mat biasI;
	//�����
	public Mat weigthOH;
	public Mat weigthOX;
	public Mat biasO;
	//��ʱ��Ԫ״̬
	public Mat weigthCH;
	public Mat weigthCX;
	public Mat biasC;
	
	//��ǰʱ��(����)��ʼ��Ϊ0
	private int time = 0;
	
	//��һ��ArrayListװ�ظ���ʱ�̵��������h �� ��Ԫ״̬c
	public ArrayList<Mat> everytimeStateOfH = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfC = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfF = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfI = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfO = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeStateOfTempC = new ArrayList<Mat>();

	// ---------
	//��һ��ArrayListװ�ظ���ʱ�̵����״̬h ,�����o ,������i ,������f ,��ʱ��Ԫ״̬tempC �����
	public ArrayList<Mat> everytimeDeltaOfH = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfO = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfI = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfF = new ArrayList<Mat>();
	public ArrayList<Mat> everytimeDeltaOfTempC = new ArrayList<Mat>();
	
	//���ź���ʱ��Ԫ��Ԫ״̬��Ȩ����ƫ�õ����ݶ�ֵ(���ݶ�ֵ!!!)
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
	 * 								����/���� -�ָ���
	 * ============================================================================
	 * */
	
	/*
	 * ��ʼ��Ȩ����ƫ�þ���
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
	 * ��ʼ��ǰ�������еĸ������Լ���Ԫ״̬����ʱ��Ԫ״̬
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
	 * ���췽��
	 * ���ֳ�ʼ�����붼������
	 * */
	public LstmLayer(int inputWidth, int stateWidth, double learningRate) {
		this.inputWidth = inputWidth;
		this.stateWidth = stateWidth;
		this.learningRate = learningRate;
		initWeight();//��ʼ��Ȩ����ƫ��
				
		//����ʱ��(����ʱ��)��(ǰ������ֵ)�������h �͵�Ԫ״̬c,��ʱ��Ԫ״̬,�����ţ������ţ������װ����
		initForwardState();
		
		//��ʼ��Ȩ�������õ��ݶ�
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
	 * ǰ������
	 * stateOfX : ��ǰʱ�̵���������
	 * �����ʱ�̵�ǰ��ֵ����װ����ArrayList
	 * */
	public void forward(Mat stateOfX) {
		this.time ++;
		
		//�����ż���
		Mat fGate = calculateGate(stateOfX, weigthFH, weigthFX, biasF, true);
		everytimeStateOfF.add(fGate);
		//�����ż���
		Mat iGate = calculateGate(stateOfX, weigthIH, weigthIX, biasI, true);
		everytimeStateOfI.add(iGate);
		//����ż���
		Mat oGate = calculateGate(stateOfX, weigthOH, weigthOX, biasO, true);
		everytimeStateOfO.add(oGate);
		//��ʱ��Ԫ״̬
		Mat tempC = calculateGate(stateOfX, weigthCH, weigthCX, biasC, false);
		everytimeStateOfTempC.add(tempC);
		//��Ԫ״̬ C = fGate .* lastC + iGate .* tempC 
		// lastC = everytimeStateOfC(time - 1);
		Mat C = Mat.plus(Mat.mulByEle(fGate, everytimeStateOfC.get(this.time - 1)), Mat.mulByEle(iGate, tempC));
		//��ʱ�̵ĵ�Ԫ״̬װ��
		everytimeStateOfC.add(C);
		
		//�������H = oGate .* C
		Mat H = Mat.mulByEle(oGate, C);
		//��ʱ�̵��������װ��
		everytimeStateOfH.add(H);
		
	}
	
	/*
	 * ͨ�õ��ż��㷽��
	 * stateOfX : ��ǰʱ�̵���������
	 * flag : ��־λ true �������� 
	 * 			    false����ʱ��Ԫ״̬
	 * */
	public Mat calculateGate(Mat stateOfX, Mat weigthH, Mat weigthX, Mat bias, boolean flag) {
		//��һʱ�̵����
		Mat lastH = everytimeStateOfH.get(time - 1);
		
		//��ʱ�̵ļ�Ȩ���� net = weigthH * lastH + weigthX * stateOfX + bias
		Mat net = Mat.plus(Mat.plus(Mat.mul(weigthH, lastH), Mat.mul(weigthX, stateOfX)), bias);
		
		//�ž���
		Mat gate = flag ? Activator.sigmoid(net) : Activator.tanh(net);
		return gate;
	}

	/*
	 * �������
	 * */
	public void backward(Mat deltaH, Mat stateOfX) {
		//�������
		calculateDelta(deltaH);
		
		//�����ݶ�
		calculateGradient(stateOfX);
		
	}
	
	/*
	 * �������
	 * */
	public void calculateDelta(Mat deltaH) {
		//��ʼ������ʱ�̵����״̬h ,�����o ,������i ,������f ,��ʱ��Ԫ״̬tempC �����
		//Mat init = Mat.zeroMat(stateWidth, 1);
		for (int i = 0; i < this.time + 1; i++) {		
			everytimeDeltaOfH.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfO.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfI.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfF.add(Mat.zeroMat(stateWidth, 1));
			everytimeDeltaOfTempC.add(Mat.zeroMat(stateWidth, 1));
		}

		//���浱ǰʱ�̵�������
		everytimeDeltaOfH.set(everytimeDeltaOfH.size() - 1, deltaH);
		
		//�������ʱ�̵ĸ��������
		for (int i = this.time; i > 0; i--) {
			//��iʱ�̵�ǰ������ֵ
			Mat fGate_i = everytimeStateOfF.get(i);
			Mat iGate_i = everytimeStateOfI.get(i);
			Mat oGate_i = everytimeStateOfO.get(i);
			Mat tempC_i = everytimeStateOfTempC.get(i);
			Mat C_i = everytimeStateOfC.get(i);
			Mat lastC = everytimeStateOfC.get(i - 1);
			//����tanh(C)
			Mat tanhC_i = Activator.tanh(C_i);
			//��ʱ��(��iʱ��)��deltaH_iֵ
			Mat deltaH_i = everytimeDeltaOfH.get(i);
			
			Mat deltaO_i = Mat.mulByEle(deltaH_i, tanhC_i, Activator.sigmoidDerivative(oGate_i));//����deltaO
			Mat deltaF_i = Mat.mulByEle(deltaH_i, oGate_i, Activator.tanhDerivative(C_i), lastC, Activator.sigmoidDerivative(fGate_i));//����deltaO
			Mat deltaI_i = Mat.mulByEle(deltaH_i, oGate_i, Activator.tanhDerivative(C_i), tempC_i, Activator.sigmoidDerivative(iGate_i));//����deltaO
			Mat deltaTempC_i = Mat.mulByEle(deltaH_i, oGate_i, Activator.tanhDerivative(C_i), iGate_i, Activator.tanhDerivative(tempC_i));
			
			//����� i-1 ʱ�� ��deltaHֵlastDeltaH
			Mat lastDeltaH = Mat.transpose(
										Mat.plus(
												Mat.mul(Mat.transpose(deltaO_i), this.weigthOH),
												Mat.mul(Mat.transpose(deltaF_i), this.weigthFH),
												Mat.mul(Mat.transpose(deltaI_i), this.weigthIH),
												Mat.mul(Mat.transpose(deltaTempC_i), this.weigthCH)
												));
					
			//����delta���ֵ
			everytimeDeltaOfH.set(i - 1, lastDeltaH);
			everytimeDeltaOfF.set(i, deltaF_i);
			everytimeDeltaOfI.set(i, deltaI_i);
			everytimeDeltaOfO.set(i, deltaO_i);
			everytimeDeltaOfTempC.set(i, deltaTempC_i);	
		}		
	}
	
	/*
	 * �����ݶ�
	 * */
	public void calculateGradient(Mat stateOfX) {
		
		for (int i = this.time; i > 0; i--) {
			Mat lastH = everytimeStateOfH.get(i - 1);
			//���㵱ǰʱ��i��W*h��B*�ݶ�
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
			
			//�ܵ��ݶ��Ǹ���ʱ���ݶ�֮��
			totalGradOfWfh = Mat.plus(totalGradOfWfh, gradOfWfh_i);
			totalGradOfWih = Mat.plus(totalGradOfWih, gradOfWih_i);
			totalGradOfWoh = Mat.plus(totalGradOfWoh, gradOfWoh_i);
			totalGradOfWch = Mat.plus(totalGradOfWch, gradOfWch_i);
			
			totalGradOfBf = Mat.plus(totalGradOfBf, gradOfBf);
			totalGradOfBi = Mat.plus(totalGradOfBi, gradOfBi);
			totalGradOfBo = Mat.plus(totalGradOfBo, gradOfBo);
			totalGradOfBc = Mat.plus(totalGradOfBc, gradOfBc);
				
		}
		//���㵱ǰʱ�̵�W*x��Ȩ��
		gradOfWfx = Mat.mul(everytimeDeltaOfF.get(everytimeDeltaOfF.size() - 1), Mat.transpose(stateOfX));//Mat.transpose(stateOfX)		
		gradOfWix = Mat.mul(everytimeDeltaOfI.get(everytimeDeltaOfI.size() - 1), Mat.transpose(stateOfX));
		gradOfWox = Mat.mul(everytimeDeltaOfO.get(everytimeDeltaOfO.size() - 1), Mat.transpose(stateOfX));
		gradOfWcx = Mat.mul(everytimeDeltaOfTempC.get(everytimeDeltaOfTempC.size() - 1), Mat.transpose(stateOfX));
	}
	
	/*
	 * ����������ݶȸ���Ȩ��
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
	 * ���ԣ��ݶȼ�����Ҫ�õ�״̬����
	 * ״̬����
	 * */
	public void resetState() {
		this.time = 0;
		initForwardState();		
	}
		
	/*
	 * ���ݼ���ѵ��
	 * */
	public void train() {
		
	}
	
	
	public static void main(String[] args) {
		
	}
}



























