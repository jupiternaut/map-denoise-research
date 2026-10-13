"""Public physical renderer only. No estimator, priors over worlds, or labels."""
import numpy as np

BRANCHES = ('none', 'partial', 'full')
FRONT = (-20., 20., -15., 15.)
LEFT = (-20., 0., -15., 15.)
RIGHT = (0., 20., -15., 15.)
REARS = {'none': None, 'partial': (-20., 36., -17., 23.),
         'full': (-100., 100., -80., 80.)}
PIXELS = np.array([(u, u + .05, v, v + .16 / 6)
                   for v in np.linspace(-.08, .08, 7)[:-1]
                   for u in np.linspace(-.2, .2, 9)[:-1]])
PIXEL_AREA = .05 * .16 / 6

def project(rect, depth, camera):
    if rect is None:
        return None
    a, b, c, d = rect
    return ((a-camera)/depth, (b-camera)/depth, c/depth, d/depth)

def intersection(a, b):
    if a is None or b is None:
        return None
    r = (max(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), min(a[3],b[3]))
    return r if r[0] < r[1] and r[2] < r[3] else None

def area_in_pixels(rect):
    if rect is None:
        return np.zeros(48)
    return (np.maximum(0, np.minimum(rect[1],PIXELS[:,1])-np.maximum(rect[0],PIXELS[:,0])) *
            np.maximum(0, np.minimum(rect[3],PIXELS[:,3])-np.maximum(rect[2],PIXELS[:,2]))) / PIXEL_AREA

def areas(depth, branch, offsets):
    """Return 3 views x 48 pixels x 3 visible material area fractions."""
    output=[]
    for center, error in zip((-80.,0.,80.), offsets):
        cam=center+error
        front=project(FRONT,depth,cam)
        rear=project(REARS[branch],720.,cam)
        output.append(np.stack((area_in_pixels(project(LEFT,depth,cam)),
                                area_in_pixels(project(RIGHT,depth,cam)),
                                area_in_pixels(rear)-area_in_pixels(intersection(rear,front))),axis=-1))
    return np.asarray(output)

def render(depth, branch, offsets, colors):
    """colors: 3 views x 3 materials, output 432 RGB components."""
    c=np.asarray(colors,dtype=float).reshape(3,3)
    gray=1-np.einsum('vpk,vk->vp',areas(depth,branch,offsets),1-c)
    return np.repeat(gray[:,:,None],3,axis=2).ravel()
