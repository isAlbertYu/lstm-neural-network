package com.albert.lstm;

class Activator {
	private Activator() {
		
	}
	
	
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(x));
	}
	
	public static double sigmoidDerivative(double x) {
		double y = sigmoid(x);
		return y * (1 - y);
	}
	
	/*
	 * 矩阵版
	 * */
	public static Mat sigmoid(Mat A) {
		double[][] res = new double[A.getRow()][A.getCol()];
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getCol(); j++) {
				res[i][j] = sigmoid(A.getElement(i, j));
			}
		}
		return new Mat(res);
	}
	/*
	 * 如果A是矩阵函数，且A的函数形式为 A = sigmoid(B)
	 * 则有求导性质 : A' = sigmoid'(B) = sigmoid(B)*(1 - sigmoid(B)) = A * (1 - A)
	 * */
	public static Mat sigmoidDerivative(Mat A) {
		double[][] res = new double[A.getRow()][A.getCol()];
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getCol(); j++) {
				res[i][j] = 1.0 - A.getElement(i, j);
			}
		}
		return Mat.mulByEle(A, new Mat(res));	
	}
	
	
	
	public static double tanh(double x) {
		/*double a = Math.exp(x);
		double b = Math.exp(-1 * x);		
		return (a - b) / (a + b);*/
		return Math.tanh(x);
	}
	
	public static double tanhDerivative(double x) {
		double y = tanh(x);
		return 1 - y * y;
	}

	/*
	 * 矩阵版
	 * */
	public static Mat tanh(Mat A) {
		double[][] res = new double[A.getRow()][A.getCol()];
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getCol(); j++) {
				res[i][j] = Math.tanh(A.getElement(i, j));
			}
		}
		return new Mat(res);
	}
	
	
	public static Mat tanhDerivative(Mat A) {
		return Mat.subs(Mat.oneMat(A.getRow(), A.getCol()), Mat.mulByEle(A, A));	
	}
	
	public static void main(String[] args) {
		double[][] am = {{1,2},{3,4},{5,6}};
		Mat A = new Mat(am);
		Mat.printMat(tanhDerivative(A));
	/*	Mat.printMat(sigmoid(A));
		System.out.println("===");
		Mat.printMat(tanh(A));*/
		
	}
	
}






