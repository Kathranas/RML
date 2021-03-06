#include <cassert>
#include <type_traits>
#include <array>
#include <cmath>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

namespace rml
{

/* A general matrix */

template<typename T, std::size_t M, std::size_t N> struct Mat
{
	union
	{
		std::array<T, M*N> arr;
		std::array<std::array<T, N>, M> arr2d;
	};

//	Mat() = default;
//	Mat(std::array<T, M*N> data) : arr{data} {}
//	Mat(std::array<std::array<T, N>, M> data) : arr2d{data} {}

	std::array<T, N>& operator[](const std::size_t i)
	{
		return arr2d[i];
	}
	const std::array<T, N>& operator[](const std::size_t i) const
	{
		return arr2d[i];
	}
	Mat& operator+=(const Mat& rhs)
	{
		for(std::size_t i=0; i<M*N; ++i)
		{
			this->arr[i] += rhs.arr[i];
		}
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		for(std::size_t i=0; i<M*N; ++i)
		{
			this->arr[i] -= rhs.arr[i];
		}
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		for(std::size_t i=0; i<M*N; ++i)
		{
			this->arr[i] *= rhs;
		}
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		for(std::size_t i=0; i<M*N; ++i)
		{
			this->arr[i] /= rhs;
		}
		return *this;
	}
};

template<typename T, std::size_t M, std::size_t N> bool operator==(const Mat<T, M, N>& lhs, const Mat<T, M, N>& rhs)
{
	for(std::size_t i=0; i<M*N; ++i)
	{
		if(lhs.arr[i] != rhs.arr[i])
			return false;
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

/* A general vector: a one dimensional matrix. */

template<typename T, std::size_t N> struct Mat<T, 1, N>
{
	union
	{
		std::array<T, N> arr;
		struct{T x, y, z, w;};
	};

	Mat() = default;
	Mat(std::array<T, N> data) : arr{data} {}

	T& operator[](const std::size_t i)
	{
		return arr[i];
	}
	const T& operator[](const std::size_t i) const
	{
		return arr[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->arr[i] += rhs.arr[i];
		}
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->arr[i] -= rhs.arr[i];
		}
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->arr[i] *= rhs;
		}
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		for(std::size_t i=0; i<N; ++i)
		{
			this->arr[i] /= rhs;
		}
		return *this;
	}
};

/* 1D vector */

template<typename T> struct Mat<T, 1, 1>
{
	union 
	{
		std::array<T, 1> arr;
		struct{T x;};
	};

	Mat() = default;
	Mat(std::array<T, 1> data) : arr{data} {}
	Mat(T data_x) : x{data_x} {}

	T& operator[](const std::size_t i)
	{
		return arr[i];
	}
	const T& operator[](const std::size_t i) const
	{
		return arr[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->x += rhs.x;
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->x -= rhs.x;
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->x *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->x /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 1>& lhs, const Mat<T, 1, 1>& rhs)
{
	return lhs.x == rhs.x;
}

/* 2D vector */

template<typename T> struct Mat<T, 1, 2>
{
	union
	{
		std::array<T, 2> arr;
		struct{T x, y;};
	};

	Mat() = default;
	Mat(std::array<T, 2> data) : arr{data} {}
	Mat(T data_x, T data_y) : x{data_x}, y{data_y} {}

	T& operator[](std::size_t i)
	{
		return arr[i];
	}
	const T& operator[](std::size_t i) const
	{
		return arr[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->x += rhs.x;
		this->y += rhs.y;
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->x -= rhs.x;
		this->y -= rhs.y;
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->x *= rhs;
		this->y *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->x /= rhs;
		this->y /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 2>& lhs, const Mat<T, 1, 2>& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;
}

/* 3D vector */

template<typename T> struct Mat<T, 1, 3>
{
	union
	{
		std::array<T, 3> arr;
		struct{T x, y, z;};
	};

	Mat() = default;
	Mat(std::array<T, 3> data) : arr{data} {}
	Mat(T data_x, T data_y, T data_z) : x{data_x}, y{data_y}, z{data_z} {}

	T& operator[](const std::size_t i)
	{
		return arr[i];
	}
	const T& operator[](const std::size_t i) const
	{
		return arr[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->x += rhs.x;
		this->y += rhs.y;
		this->z += rhs.z;
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->x -= rhs.x;
		this->y -= rhs.y;
		this->z -= rhs.z;
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->x *= rhs;
		this->y *= rhs;
		this->z *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->x /= rhs;
		this->y /= rhs;
		this->z /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 3>& lhs, const Mat<T, 1, 3>& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

/* 4D vector */

template<typename T> struct Mat<T, 1, 4>
{
	union
	{
		std::array<T, 4> arr;
		struct{T x, y, z, w;};
	};

	Mat() = default;
	Mat(std::array<T, 4> data) : arr{data} {}
	Mat(T data_x, T data_y, T data_z, T data_w) : x{data_x}, y{data_y}, z{data_z}, w{data_w} {}

	T& operator[](const std::size_t& i)
	{
		return arr[i];
	}
	const T& operator[](const std::size_t& i) const
	{
		return arr[i];
	}

	Mat& operator+=(const Mat& rhs)
	{
		this->x += rhs.x;
		this->y += rhs.y;
		this->z += rhs.z;
		this->w += rhs.w;
		return *this;
	}
	Mat& operator-=(const Mat& rhs)
	{
		this->x -= rhs.x;
		this->y -= rhs.y;
		this->z -= rhs.z;
		this->w -= rhs.w;
		return *this;
	}
	Mat& operator*=(const T& rhs)
	{
		this->x *= rhs;
		this->y *= rhs;
		this->z *= rhs;
		this->w *= rhs;
		return *this;
	}
	Mat& operator/=(const T& rhs)
	{
		this->x /= rhs;
		this->y /= rhs;
		this->z /= rhs;
		this->w /= rhs;
		return *this;
	}
};
template<typename T> bool inline operator==(const Mat<T, 1, 4>& lhs, const Mat<T, 1, 4>& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}


template<typename T, std::size_t N> T dot(const std::array<T, N>& a, const std::array<T, N>& b)
{
	T total = 0;

	for(std::size_t i=0; i<N; ++i)
	{
		total += a[i] * b[i];
	}
	return total;
}

template<typename T, std::size_t N> T inline dot(const Mat<T, 1, N>& a, const Mat<T, 1, N>& b)
{
	return dot(a.arr, b.arr);
}

template<typename T, std::size_t N> T inline abs_squared(const std::array<T, N>& a)
{
	return dot(a, a);
}

template<typename T, std::size_t N> T inline abs(const std::array<T, N>& a)
{
	return std::sqrt(abs_squared(a));
}

template<typename T, std::size_t N> T inline abs(const Mat<T, 1, N>& a)
{
	return abs(a);
}

template<typename T, std::size_t N> Mat<T, 1, N> direction(const Mat<T, 1, N>& a)
{
	return a / abs(a.arr);
}

template<typename T> Mat<T, 1, 3> cross(const Mat<T, 1, 3>& a, const Mat<T, 1, 3>& b) // only possible in 3D
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
		ret[i] = dot(a[i], b.arr);
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
	Mat<T, 1, 3> u = v / rml::abs(v.arr);

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

template<typename T> T to_radians(T degrees)
{
	return degrees * M_PI / 180;
}

template<typename T> T to_degrees(T radians)
{
	return radians * 180 / M_PI;
}

template<typename T> Mat<T, 4, 4> perspective(float fov, float aratio, float n, float f)
{

	float t = std::tan(to_radians(fov) / 2.0f) * n;
	float b = -t;
	float r = t * aratio;
	float l = -r;

	Mat<T, 4, 4> mat
	{
		2*n/(r - l),     0,               0,                0,
		0,               2*n / (t - b),   0,                0,
	    (r + l)/(r - l), (t + b)/(t - b), -(f + n)/(f - n), -1,
		0,               0,               -2*f*n / (f - n), 0
	};

	return mat;
}

template<typename T> Mat<T, 4, 4> look_at(const Mat<T, 1, 3>& p, const Mat<T, 1, 3>& t, const Mat<T, 1, 3>& uu)
{
	Mat<T, 1, 3> d = direction(p - t);
	Mat<T, 1, 3> r = direction(cross(uu, d));
	Mat<T, 1, 3> u = cross(d, r);

	Mat<T, 4, 4> temp
	{
		1, 0, 0, -p.x,
		0, 1, 0, -p.y,
		0, 0, 1, -p.z,
		0, 0, 0, 1
	};

	Mat<T, 4, 4> look
	{
		r.x,  r.y,  r.z,  0,
	    u.x,  u.y,  u.z,  0,
		d.x,  d.y,  d.z,  0,
		0,    0,    0,    1
	};

	return rml::multiply(look, temp);
}

} // namespace rml
