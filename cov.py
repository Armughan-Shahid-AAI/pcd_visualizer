import numpy as np
import matplotlib.pyplot as plt


def get_principal_axes_bbox2d(x, y) -> tuple:
    r"""
        Returns 2 points defining a 2d bounding box fitted using principal components.
    Args:
        x: points coord
        y: points coord
    Returns:
        (p1, p2): tuple
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    p1, p2 = find_bbox2d(x, y, evec_1=(x_v1, y_v1), evec_2=(x_v2, y_v2))
    plot_axes(x, y, (x_v1, y_v1), (x_v2, y_v2), p1, p2)

    return p1, p2


def get_principal_axes_bbox3d(x, y, z, axes: str = 'xy'):
    r"""
            Returns 2 points defining a 2d bounding box fitted using principal components obtained from 3d data.
    Args:
        x: points coord
        y: points coord
        z: points coord
        axes: projection axes = ['xy', 'yz' or 'xz']
    Returns:
        (p1, p2): tuple
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)

    coords = np.vstack([x, y, z])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]

    if axes == "xy":
        x_v1, y_v1 = evecs[:2, sort_indices[0]]
        x_v2, y_v2 = evecs[:2, sort_indices[1]]
        px, py = x, y
    elif axes == "yz":
        x_v1, y_v1 = evecs[1:, sort_indices[0]]
        x_v2, y_v2 = evecs[1:, sort_indices[1]]
        px, py = y, z
    elif axes == "xz":
        x_v1, y_v1 = evecs[[0, 2], sort_indices[0]]
        x_v2, y_v2 = evecs[[0, 2], sort_indices[1]]
        px, py = x, z
    else:
        raise ValueError("plot arg can take following values: ['xy', 'xz', 'yz']")

    # projecting on new axes and finding boundaries
    p1, p2 = find_bbox2d(px, py, evec_1=(x_v1, y_v1), evec_2=(x_v2, y_v2))
    plot_axes(px, py, (x_v1, y_v1), (x_v2, y_v2), p1, p2)
    return p1, p2


def find_bbox2d(x, y, evec_1: tuple, evec_2: tuple, keep_coord: bool = True) -> tuple:
    import time
    s = time.perf_counter()
    proj_x_v1, proj_y_v1 = proj_on_eigvec(x, y, vec=evec_1)  # x?
    proj_x_v2, proj_y_v2 = proj_on_eigvec(x, y, vec=evec_2)  # y?
    print("Elapsed: {}".format(time.perf_counter() - s))

    if keep_coord:
        # keep coordinate system, principal axes for reference
        l = min(proj_x_v1)
        r = max(proj_x_v1)
        u = min(proj_y_v2)
        d = max(proj_y_v2)
        return (l, u), (r, d)
    else:
        # in pca coordinates
        raise NotImplementedError()


def proj_on_eigvec(x, y, vec: tuple) -> tuple:
    proj_x, proj_y = [], []
    for point_x, point_y in zip(x, y):
        x = np.array([point_x, point_y])
        u = np.array([0.0, 0.0])
        v = np.array(vec)

        n = v - u
        n /= np.linalg.norm(n, 2)

        P = u + n * np.dot(x - u, n)
        proj_x.append(P[0])
        proj_y.append(P[1])
    return proj_x, proj_y


def plot_axes(x, y, vec1, vec2, p1, p2, scale=1):
    x_v1, y_v1 = vec1
    x_v2, y_v2 = vec2

    plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
             [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
    plt.plot([x_v2 * -scale, x_v2 * scale],
             [y_v2 * -scale, y_v2 * scale], color='blue')
    plt.plot(x, y, 'k.')
    plt.axis('equal')
    # plt.gca().invert_yaxis()

    plt.axvline(x=p1[0])
    plt.axvline(x=p2[0])
    plt.axhline(y=p2[1])
    plt.axhline(y=p1[1])
    plt.show()