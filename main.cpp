#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

double errRel(const Vector2d x, const Vector2d x_ex)
{
	return (x - x_ex).norm() / x_ex.norm();
}



int main()
{
	Vector2d x_ex = -Vector2d::Ones(2);
	
	// Matrici
	
	vector<Matrix2d> A_matrix = {
		(Matrix2d() << 5.547001962252291e-01, -3.770900990025203e-02, 
		               -9.992887623566787e-01, 8.320502943378437e-01).finished(),
		(Matrix2d() << 5.547001962252291e-01, -5.540607316466765e-01,
		               -8.324762492991313e-01, 8.320502943378437e-01).finished(),
		(Matrix2d() << 5.547001962252291e-01, -5.547001955851905e-01,
		               -8.320502947645361e-01, 8.320502943378437e-01).finished()
	};
	
	// Vettori
	
	vector<Vector2d> b_vector = {
		Vector2d(-5.169911863249772e-01, 1.672384680188350e-01),
		Vector2d(-6.394645785530173e-04, 4.259549612877223e-04),
		Vector2d(-6.400391328043042e-10, 4.266924591433963e-10)
	};
	
	
	for (unsigned int i = 0; i < A_matrix.size(); i++)
	{
		Matrix2d A = A_matrix[i];
		Vector2d b = b_vector[i];
		
		cout << "\n";
		cout << "Sistema " << i+1 << endl;
		
		// PALU
		PartialPivLU<Matrix2d> palu(A);
		Vector2d x_palu = palu.solve(b);
		cout << "Soluzione (PALU): x = " << x_palu.transpose() << endl;
		cout << "Errore relativo (PALU): err = " << errRel(x_palu, x_ex) << endl;
		
		
		// QR
		HouseholderQR<Matrix2d> qr(A);
		Vector2d x_qr = qr.solve(b);
		cout << "Soluzione (QR): x = " << x_qr.transpose() << endl;
		cout << "Errore relativo (QR): err = " << errRel(x_qr, x_ex) << endl;
		
	}
	
	cout << "\n";
	
	return 0;
}




