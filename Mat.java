package com.albert.lstm;

import java.util.ArrayList;
import java.util.Arrays;

class Mat {
	private double[][] mat;
	
	/*
	 * 构造方法族
	 * */
	private Mat() {//不让你以这种没有意义的方式初始化
	}
	
	public Mat(int row, int col) {
		mat = new double[row][col];
	}
	
	public Mat(double[][] array) {
		double[][] copyOfArray = new double[array.length][array[0].length];
		//double[][] arr = (double[][]) array.clone();//这是浅拷贝，不可用
		for (int i = 0; i < array.length; i++) {
			System.arraycopy(array[i], 0, copyOfArray[i], 0, array[0].length);
		}
		this.mat = copyOfArray;	
		//this.mat = Arrays.copyOf(original, newLength);
	}

	/*
	 * 返回一个指定规格的零矩阵
	 * */
	public static Mat zeroMat(int row, int col) {
		double[][] array = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				array[i][j] = 0;
			}
		}
		return new Mat(array);
	}
	
	/*
	 * 返回一个指定规格的全一矩阵
	 * */
	public static Mat oneMat(int row, int col) {
		double[][] array = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				array[i][j] = 1;
			}
		}
		return new Mat(array);
	}
	
	/*
	 * 获取矩阵中的指定位置处的元素
	 * */
	public double getElement(int row, int col) {
		return this.mat[row][col];
	}
	
	/*
	 * 设置矩阵的指定位置的元素为指定的元素
	 * */
	public void setElement(int row, int col, double ele) {
		this.mat[row][col] = ele;
	}
	
	/*
	 * 获取矩阵的行数
	 * */
	public int getRow() {
		return this.mat.length;
	}
	
	/*
	 * 获取矩阵的列数
	 * */
	public int getCol() {
		return this.mat[0].length;
	}

	
	/*打印矩阵*/
	public static void printMat(Mat A) {
		if (A != null) {
			for (int i = 0; i < A.getRow(); i++) {
				System.out.println(Arrays.toString(A.mat[i]));
			}
		} else {
			System.out.println("打印结果为 空矩阵");
		}
		
	}
	
	/*
	 * 矩阵相加,返回和矩阵
	 * */
	public static Mat plus(Mat A, Mat B){
		if (A.getRow() != B.getRow() || A.getCol() != B.getCol()) {
/*			System.out.println("矩阵的规格不一致，无法相加");
			return null;*/
			throw new MatOperatingException("矩阵的规格不一致，无法相加");
		} else {
			Mat C = new Mat(A.mat);
			for (int i = 0; i < C.getRow(); i++) {
				for (int j = 0; j < C.getCol(); j++) {
					C.mat[i][j] += B.mat[i][j];
				}
			}
			return C;
		}		
	}
	
	/*
	 * 矩阵相加,返回和矩阵
	 * 可变参数
	 * */
	public static Mat plus(Mat...args) {
		int row = args[0].getRow();
		int col = args[0].getCol();
		for (int i = 1; i < args.length; i++) {
			if (args[i].getRow() != row || args[i].getCol() != col) {
/*				System.out.println("第" + i + "个矩阵的规格不合法，无法相加");
				return null;*/
				throw new MatOperatingException("第" + i + "个矩阵的规格不合法，无法相加");
			}
		}
		Mat C = Mat.zeroMat(row, col);
		for (int k = 0; k < args.length; k++) {
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					C.mat[i][j] += args[k].mat[i][j];
				}
			}
		}		
		return C;	
	}
	
	
	/*
	 * 矩阵相减, 返回差矩阵
	 * */
	public static Mat subs(Mat A, Mat B) {
		if (A.getRow() != B.getRow() || A.getCol() != B.getCol()) {
			throw new MatOperatingException("矩阵的规格不一致，无法相减");
		} else {
			Mat C = new Mat(A.mat);
			for (int i = 0; i < C.getRow(); i++) {
				for (int j = 0; j < C.getCol(); j++) {
					C.mat[i][j] -= B.mat[i][j];
				}
			}
			return C;
		}		
	}
	
	/*
	 * 矩阵相乘, 返回矩阵
	 * */
	public static Mat mul(Mat A, Mat B) {
		if (A.getCol() != B.getRow()) {
			throw new MatOperatingException("左矩阵的列数 不等于 右矩阵的行数， 不能相乘");
		} else {
			double[][] result = new double[A.getRow()][B.getCol()];
	        for(int i = 0 ; i < A.getRow() ; i++ ) {
	        	for(int j = 0 ; j < B.getCol() ; j++) {
	            	 for(int k = 0 ; k < B.getRow() ; k++ ){  
	                    result[i][j] += A.mat[i][k] * B.mat[k][j];    
	                }  
	            }  
	        }
	        return new Mat(result);             
		}
	}
	
/*	
	 * 矩阵 按元素乘
	 */
	public static Mat mulByEle(Mat A, Mat B) {
		if (A.getRow() != B.getRow() || A.getCol() != B.getCol()) {
			throw new MatOperatingException("矩阵的规格不一致，无法按元素乘");
		} else {
			double[][] result = new double[A.getRow()][A.getCol()];
	        for(int i = 0 ; i < A.getRow() ; i++ ) {
	        	for(int j = 0 ; j < A.getCol() ; j++) {
	        		result[i][j] = A.mat[i][j] * B.mat[i][j];  
	            }
	        }
	        return new Mat(result);
		}
	}
	
	/*
	 * 矩阵 按元素乘
	 * 可变参数
	 * */
	public static Mat mulByEle(Mat...args) {
		int row = args[0].getRow();
		int col = args[0].getCol();
		for (int i = 1; i < args.length; i++) {
			if (args[i].getRow() != row || args[i].getCol() != col) {
				throw new MatOperatingException("第" + i + "个矩阵的规格不合法，无法按元素乘");
			}
		}
		double[][] result = new double[row][col];
        for(int i = 0 ; i < row ; i++ ) {
        	for(int j = 0 ; j < col ; j++) {
        		result[i][j] = 1.0;  
        		for (int k = 0; k < args.length; k++) {
					result[i][j] *=  args[k].mat[i][j];
				}    		
            }
        }
        return new Mat(result);
	}
	
	/*
	 * 矩阵转置
	 * */
	public static Mat transpose(Mat A) {
		double[][] res = new double[A.getCol()][A.getRow()];
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getCol(); j++) {
				res[j][i] = A.mat[i][j];
			}
		}
		return new Mat(res);
	}
	
	/*
	 * 矩阵数乘
	 * */
	public static Mat scalMul(double num, Mat A) {
		double[][] res = new double[A.getRow()][A.getCol()];
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getCol(); j++) {
				res[i][j] = num * A.mat[i][j];
			}
		}
		return new Mat(res);
	}
	
	/*
	 * 矩阵除以一个数
	 * */
	public static Mat divNum(Mat A, double num) {
		double[][] res = new double[A.getRow()][A.getCol()];
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getCol(); j++) {
				res[i][j] = A.mat[i][j]/num;
			}
		}
		return new Mat(res);
	}
	
	
	/*
	 * 返回矩阵的某一行，行矩阵
	 * */
	public Mat getRowMat(int row) {
		double[][] res = new double[1][this.getCol()];
		res[0] = this.mat[row];
		return new Mat(res);
	}
	
	/*
	 * 返回矩阵的某一列，列矩阵
	 * */
	public Mat getColMat(int col) {
		return transpose(transpose(this).getRowMat(col));
	}
	
	
	public static void main(String[] args) {
		double[][] am = {{1,2,3},{4,5,6},{1,4,5},{2,1,4}};
		Mat A = new Mat(am);
				
		double[][] bm = {{0,1,2},{6,7,8},{2,3,1},{2,3,1}};
		Mat B = new Mat(bm);
		Mat xx = Mat.plus(A, B);
		
		System.out.println("=+++=");
		printMat(B.getRowMat(2));
		System.out.println("=+++=");
		printMat(B.getColMat(1));
		
		printMat(new Mat(new double[][]{{1,2,3},{4,5,6}}));
		
		double[][] cm = {{6,7,8},{8,9,10},{11,23,12},{6,7,8}};
		Mat C = new Mat(cm);
		
		double[][] dm = {{2.8,1.1,8.1},{8.8,9.1,110},{121,213,122},{1,4,5}};
		Mat D = new Mat(dm);
		
/*		double[][] dm = {{1},{2},{7}};
		Mat D = new Mat(dm);*/
		
/*		printMat(plus(A,B));
		System.out.println("===");
		
		printMat(mul(A,C));		
		System.out.println("=A=");
		printMat(A);
		System.out.println("A.*B=");
		printMat(mulByEle(A,B));
		
		System.out.println("++++++++++++++++");
		int[][] abc = {{1,2,3},{3,4,5},{5,6,7}};
		System.out.println(Arrays.toString(abc[0]));*/
		
	/*	printMat(mulByEle(A,B,C,D));
		System.out.println("==");
		printMat(transpose(mulByEle(A,B,C,D)));
		System.out.println("=...=");
		
		double[][] xm = {{1,2,3}};
		Mat x = new Mat(xm);
		printMat(x);
		System.out.println("=----=");
		printMat(transpose(x));
		ArrayList<String> strArr = new ArrayList<String>();
		System.out.println(strArr);
		strArr.add("asd");
		System.out.println(strArr);
		strArr.clear();
		System.out.println(strArr);
		
		System.out.println("=----=");*/
		
		
	}

}


class MatOperatingException extends RuntimeException{
    public MatOperatingException(){
        super();
    }
    public MatOperatingException(String msg){
        super(msg);
    }
}

















