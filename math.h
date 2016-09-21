#include <cassert>
#include <type_traits>
#include <array>
#include <cmath>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace rml
{

template<typename T, std::size_t M, std::size_t N> struct Mat
{
	std::array<std::array<T, N>, M> data;

	std::array<T, N>& operator[](std::size_t i)
	{
		return this->data[i];
	}
	const std::array<T, N>& operator[](const std::size_t i) const
	{
		return this->data[i];
	}
	Mat& operator+=(const Mat& rhs)
	{
		for(std::size_t i=0; i<M; ++i)
		{
			for(std::size_t j=0; i<N; ++j)
			{
				this->data[i][j] += rhs.data[i][j];
			}
		}
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		for(std::size_t i=0; i<M; ++i)
		{
			for(std::size_t j=0; j<N; ++j)
			{
				this->data[i][j] -= rhs.data[i][j];
			}
		}
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		for(std::size_t i=0; i<M; ++i)
		{
			for(std::size_t j=0; j<N; ++i)
			{
				this->data[i][j] *= rhs;
			}
		}
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		for(std::size_t i=0; i<M; ++i)
		{
			for(std::size_t j=0; j<N; ++j)
			{
				this->data[i][j] /= rhs;
			}
		}
		return *this;
	}
};

template<typename T, std::size_t M, std::size_t N> bool operator==(const Mat<T, M, N>& lhs, const Mat<T, M, N>& rhs)
{
	for(std::size_t i=0; i<M; ++i)
	{
		for(std::size_t j=0; j<N; ++j)
		{
			if(lhs[i][j] != rhs[i][j])
				return false;
		}
	}
	return true;
}
template<typename T, std::size_t M, std::size_t N> bool operator!=(const Mat<T, M, N>& lhs, const Mat<T, M, N>& rhs)
{
	return !operator==(lhs, rhs);
}
template<typename T, std::size_t M, std::size_t N> Mat<T, M, N> operator+(Mat<T, M, N> lhs, const Mat<T, M, N>& rhs)
{
	lhs += rhs;
	return lhs;
}
template<typename T, std::size_t M, std::size_t N> Mat<T, M, N> operator-(Mat<T, M, N> lhs, const Mat<T, M, N>& rhs)
{
	lhs -= rhs;
	return lhs;
}
template<typename T, std::size_t M, std::size_t N> Mat<T, M, N> operator*(Mat<T, M, N> lhs, const T& rhs)
{
	lhs *= rhs;
	return lhs;
}
template<typename T, std::size_t M, std::size_t N> Mat<T, M, N> operator/(Mat<T, M, N> lhs, const T& rhs)
{
	lhs /= rhs;
	return lhs;
}
template<typename T, std::size_t M, std::size_t N> Mat<T, M, N> operator*(T lhs, Mat<T, M, N> rhs)
{
	rhs *= lhs;
	return rhs;
}

template<typename T, std::size_t N> struct Mat<T, 1, N>
{
	union
	{
		std::array<T, N> data;
		struct{T x, y, z, w;};
	};

	T& operator[](const std::size_t& i)
	{
		return data[i];
	}
	const T& operator[](const std::size_t& i) const
	{
		return data[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->data[i] += rhs[i];
		}
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->data[i] -= rhs[i];
		}
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->data[i] *= rhs;
		}
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->data[i] /= rhs;
		}
		return *this;
	}
};
template<typename T, std::size_t N> bool operator==(const Mat<T, 1, N>& lhs, const Mat<T, 1, N>& rhs)
{
	for(std::size_t i=0; i<N; ++i)
	{
		if(lhs[i] != rhs[i])
			return false;
	}
	return true;
}

template<typename T> struct Mat<T, 1, 1>
{
	union 
	{
		std::array<T, 1> data;
		struct{T x;};
	};

	T& operator[](const std::size_t& i)
	{
		return data[i];
	}
	const T& operator[](const std::size_t& i) const
	{
		return data[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->data[0] += rhs[0];
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->data[0] -= rhs[0];
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->data[0] *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->data[0] /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 1>& lhs, const Mat<T, 1, 1>& rhs)
{
	return lhs[0] == rhs[0];
}

template<typename T> struct Mat<T, 1, 2>
{
	union
	{
		std::array<T, 2> data;
		struct{T x, y;};
	};

	T& operator[](const std::size_t& i)
	{
		return data[i];
	}
	const T& operator[](const std::size_t& i) const
	{
		return data[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->data[0] += rhs[0];
		this->data[1] += rhs[1];
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->data[0] -= rhs[0];
		this->data[1] -= rhs[1];
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->data[0] *= rhs;
		this->data[1] *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->data[0] /= rhs;
		this->data[1] /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 2>& lhs, const Mat<T, 1, 2>& rhs)
{
	return lhs[0] == rhs[0] && lhs[1] == rhs[1];
}

template<typename T> struct Mat<T, 1, 3>
{
	union
	{
		std::array<T, 3> data;
		struct{T x, y, z;};
	};

	T& operator[](const std::size_t& i)
	{
		return data[i];
	}
	const T& operator[](const std::size_t& i) const
	{
		return data[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->data[0] += rhs[0];
		this->data[1] += rhs[1];
		this->data[2] += rhs[2];
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->data[0] -= rhs[0];
		this->data[1] -= rhs[1];
		this->data[2] -= rhs[2];
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->data[0] *= rhs;
		this->data[1] *= rhs;
		this->data[2] *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->data[0] /= rhs;
		this->data[1] /= rhs;
		this->data[2] /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 3>& lhs, const Mat<T, 1, 3>& rhs)
{
	return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
}

template<typename T> struct Mat<T, 1, 4>
{
	union
	{
		std::array<T, 4> data;
		struct{T x, y, z, w;};
	};

	T& operator[](const std::size_t& i)
	{
		return data[i];
	}
	const T& operator[](const std::size_t& i) const
	{
		return data[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->data[0] += rhs[0];
		this->data[1] += rhs[1];
		this->data[2] += rhs[2];
		this->data[3] += rhs[3];
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->data[0] -= rhs[0];
		this->data[1] -= rhs[1];
		this->data[2] -= rhs[2];
		this->data[3] -= rhs[3];
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->data[0] *= rhs;
		this->data[1] *= rhs;
		this->data[2] *= rhs;
		this->data[3] *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->data[0] /= rhs;
		this->data[1] /= rhs;
		this->data[2] /= rhs;
		this->data[3] /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 4>& lhs, const Mat<T, 1, 4>& rhs)
{
	return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3];
}

template<typename T, std::size_t N> T dot(const std::array<T, N>& a, const std::array<T, N>& b)
{
	T total = 0;

	for(std::size_t i=0; i<N; ++i)
	{
		total += (a[i] * b[i]);
	}
	return total;
}

template<typename T, std::size_t N> T inline abs_squared(const std::array<T, N>& a)
{
	return dot(a, a);
}

template<typename T, std::size_t N> T inline abs(const std::array<T, N>& a)
{
	return std::sqrt(abs_squared(a));
}

template<typename T> Mat<T, 1, 3> cross(const Mat<T, 1, 3>& a, const Mat<T, 1, 3>& b) // only defined for 3D
{
	return {(a.y*b.z - a.z*b.y), (a.z*b.x - a.x*b.z), (a.x*b.y - a.y*b.x)};
}

typedef Mat<float, 1, 2> Vec2;
typedef Mat<float, 1, 3> Vec3;
typedef Mat<float, 1, 4> Vec4;

static_assert(std::is_pod<Vec2>(), "VEC2 IS NOT A POD TYPE");
static_assert(std::is_pod<Vec3>(), "VEC3 IS NOT A POD TYPE");
static_assert(std::is_pod<Vec4>(), "VEC4 IS NOT A POD TYPE");

static_assert(sizeof(Vec2) == sizeof(float[2]), "VEC2 IS PADDED");
static_assert(sizeof(Vec3) == sizeof(float[3]), "VEC3 IS PADDED");
static_assert(sizeof(Vec4) == sizeof(float[4]), "VEC4 IS PADDED");

template<typename T, std::size_t M> Mat<T, M, M> identity()
{
	rml::Mat<T, M, M> identity{};

	for(std::size_t i=0; i<M; ++i)
	{
		identity[i][i] = 1;
	}
	return identity;
}

template<typename T, std::size_t M, std::size_t N> Mat<T, N, M> transpose(const Mat<T, M, N>& a)
{
	Mat<T, N, M> b;
	for(std::size_t i=0; i<M; ++i)
	{
		for(std::size_t j=0; j<N; ++j)
		{
			b[j][i] = a[i][j];
		}
	}
	return b;
}

// Mat4x2 * Mat2x3 = Mat4x3
// MatMxN * MatNxP = MatMxP
// cols a == rows b; 2 == 2 
template<typename T, std::size_t M, std::size_t N, std::size_t P> Mat<T, M, P> multiply(const Mat<T, M, N>& a, const Mat<T, N, P>& b)
{
	Mat<T, P, N> bt = transpose(b);
	Mat<T, M, P> ret;

	for(std::size_t i=0; i<M; ++i)
	{
		for(std::size_t j=0; j<P; ++j)
		{
			ret[i][j] = dot(a[i], bt[j]);
		}
	}
	return ret;
}

// MatMxN * MatNx1 = MatMx1
template<typename T, std::size_t M, std::size_t N> Mat<T, 1, M> multiply(const Mat<T, M, N>& a, const Mat<T, 1, N>& b)
{
	Mat<T, 1, M> ret;

	for(std::size_t i=0; i<M; ++i)
	{
		ret[i] = dot(a[i], b.data);
	}
	return ret;
}

template<typename T> void translate(Mat<T, 4, 4>& trans, const Mat<T, 1, 3>& v)
{
	trans[0][3] += v.x;
	trans[1][3] += v.y;
	trans[2][3] += v.z;
}

template<typename T> void scale(Mat<T, 4, 4>& trans, const Mat<T, 1, 3>& v)
{
	Mat<T, 4, 4> m = identity<T, 4>();

	m[0][0] = v.x;
	m[1][1] = v.y;
	m[2][2] = v.z;

	trans = multiply(m, trans);
}


template<typename T> void rotate(Mat<T, 4, 4>& trans, const float angle, const Mat<T, 1, 3>& v)
{
	float c = std::cos(angle);
	float s = std::sin(angle);
	Mat<T, 1, 3> u = v / rml::abs(v.data);

	Mat<T, 4, 4> m = identity<T, 4>();

	m[0][0] = (c + u.x * u.x * (1 - c));
	m[0][1] = (u.x * u.y * (1 - c) - u.z * s);
	m[0][2] = (u.x * u.z * (1 - c) + u.y * s);
	m[1][0] = (u.y * u.x * (1 - c) + u.z * s);
	m[1][1] = (c + u.y * u.y * (1 - c));
	m[1][2] = (u.y * u.z * (1 - c) - u.x * s);
	m[2][0] = (u.z * u.z * (1 - c) - u.y * s);
	m[2][1] = (u.z * u.y * (1 - c) + u.x * s);
	m[2][2] = (c + u.z * u.z * (1 - c));

	trans = multiply(m, trans);
}

float to_radians(float degrees)
{
	return degrees * M_PI / 180;
}

float to_degrees(float radians)
{
	return radians * 180 / M_PI;
}

template<typename T>Mat<T, 4, 4> perspective(float fov, float aratio, float n, float f)
{
	Mat<T, 4, 4> mat{};

	float s = 1.0f / std::tan(to_radians(fov) / 2.0f);

	mat[0][0] = s;
	mat[1][1] = s;
	mat[2][2] = - f / (f - n);
	mat[2][3] = -1;
	mat[3][2] = -(f*n) / (f - n);
	return mat;
}

} // namespace rml
