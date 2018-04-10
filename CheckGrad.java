package com.albert.lstm;

public class CheckGrad {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//检验数据集
		Mat checkXMat = new Mat(new double[][]{{1,2,3},{2,3,4}}); 
		
		LstmLayer ls = new LstmLayer(3, 2, 0.001);

		ls.forward(Mat.transpose(checkXMat.getRowMat(0)));
		ls.forward(Mat.transpose(checkXMat.getRowMat(1)));
		
		ls.backward(Mat.oneMat(ls.stateWidth, 1), Mat.transpose(checkXMat.getRowMat(1)));
		final double epsilon = 0.0001;
		for (int i = 0; i < ls.weigthFH.getRow(); i++) {
			for (int j = 0; j < ls.weigthFH.getCol(); j++) {
				ls.weigthFH.setElement(i, j, ls.weigthFH.getElement(i, j) + epsilon);
				ls.resetState();/**状态重置**/
				ls.forward(Mat.transpose(checkXMat.getRowMat(0)));
				ls.forward(Mat.transpose(checkXMat.getRowMat(1)));
				
				/*设计了一个误差函数(对所有时间节点的输出向量h求和)*/
				double errorSum1 = 0;
				for (int k = 0; k < ls.stateWidth; k++) {
					errorSum1 = errorSum1 + ls.everytimeStateOfH.get(ls.everytimeStateOfH.size() - 1).getElement(k, 0);
				}
				/*函数结束*/				
				ls.weigthFH.setElement(i, j, ls.weigthFH.getElement(i, j) - 2 * epsilon);
				ls.resetState();/**状态重置**/
				ls.forward(Mat.transpose(checkXMat.getRowMat(0)));
				ls.forward(Mat.transpose(checkXMat.getRowMat(1)));

				/*设计了一个误差函数(对当前(最新)时间节点的输出向量h求和)*/
				double errorSum2 = 0;
				for (int k = 0; k < ls.stateWidth; k++) {
					errorSum2 = errorSum2 + ls.everytimeStateOfH.get(ls.everytimeStateOfH.size() - 1).getElement(k, 0);
				}			
				/*函数结束*/
				//计算期望的梯度
				double expectGrad = (errorSum2 - errorSum1) / (2 * epsilon);
				//将权重Wfh复原
				ls.weigthFH.setElement(i, j, ls.weigthFH.getElement(i, j) + epsilon);
				//查看期望梯度与实际梯度
				System.out.println("期望梯度 ： " + expectGrad);
				System.out.println("实际梯度 ： " + ls.totalGradOfWfh.getElement(i, j));			
				System.out.println("--------");
			}
		}
		
		
		
	}

}
