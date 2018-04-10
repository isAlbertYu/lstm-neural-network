package com.albert.lstm;

public class CheckGrad {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//�������ݼ�
		Mat checkXMat = new Mat(new double[][]{{1,2,3},{2,3,4}}); 
		
		LstmLayer ls = new LstmLayer(3, 2, 0.001);

		ls.forward(Mat.transpose(checkXMat.getRowMat(0)));
		ls.forward(Mat.transpose(checkXMat.getRowMat(1)));
		
		ls.backward(Mat.oneMat(ls.stateWidth, 1), Mat.transpose(checkXMat.getRowMat(1)));
		final double epsilon = 0.0001;
		for (int i = 0; i < ls.weigthFH.getRow(); i++) {
			for (int j = 0; j < ls.weigthFH.getCol(); j++) {
				ls.weigthFH.setElement(i, j, ls.weigthFH.getElement(i, j) + epsilon);
				ls.resetState();/**״̬����**/
				ls.forward(Mat.transpose(checkXMat.getRowMat(0)));
				ls.forward(Mat.transpose(checkXMat.getRowMat(1)));
				
				/*�����һ������(������ʱ��ڵ���������h���)*/
				double errorSum1 = 0;
				for (int k = 0; k < ls.stateWidth; k++) {
					errorSum1 = errorSum1 + ls.everytimeStateOfH.get(ls.everytimeStateOfH.size() - 1).getElement(k, 0);
				}
				/*��������*/				
				ls.weigthFH.setElement(i, j, ls.weigthFH.getElement(i, j) - 2 * epsilon);
				ls.resetState();/**״̬����**/
				ls.forward(Mat.transpose(checkXMat.getRowMat(0)));
				ls.forward(Mat.transpose(checkXMat.getRowMat(1)));

				/*�����һ������(�Ե�ǰ(����)ʱ��ڵ���������h���)*/
				double errorSum2 = 0;
				for (int k = 0; k < ls.stateWidth; k++) {
					errorSum2 = errorSum2 + ls.everytimeStateOfH.get(ls.everytimeStateOfH.size() - 1).getElement(k, 0);
				}			
				/*��������*/
				//�����������ݶ�
				double expectGrad = (errorSum2 - errorSum1) / (2 * epsilon);
				//��Ȩ��Wfh��ԭ
				ls.weigthFH.setElement(i, j, ls.weigthFH.getElement(i, j) + epsilon);
				//�鿴�����ݶ���ʵ���ݶ�
				System.out.println("�����ݶ� �� " + expectGrad);
				System.out.println("ʵ���ݶ� �� " + ls.totalGradOfWfh.getElement(i, j));			
				System.out.println("--------");
			}
		}
		
		
		
	}

}
