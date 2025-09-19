import torch
import pandas as pd
import os


class Symmetry:
    """
    Class for handling symmetry operations in 3D reconstruction.

    Loads symmetry definitions from a CSV file and generates rotation and reflection matrices
    for use in symmetrizing 3D volumes.

    Attributes:
        symmetry (str): Name of the symmetry group.
        rotations (pd.DataFrame): DataFrame containing symmetry operations.
        rr (torch.Tensor): Rotation matrices for symmetry operations.
        ll (torch.Tensor): Left transformation matrices for symmetry operations.
    """
    def __init__(self, symmetry: str, symmetry_dir: str):
        """
        Args:
            symmetry (str): Name of the symmetry group.
            symmetry_dir (str): Directory containing the symmetry CSV file.
        """
        self.symmetry = symmetry
        full_path = os.path.join(symmetry_dir, f"{self.symmetry}.csv")
        self.rotations = pd.read_csv(full_path)
        self.rr = None
        self.ll = None

    def create_rotation_matrices(self):
        """
        Create rotation and reflection matrices based on the loaded symmetry operations.

        Populates self.rr and self.ll with the corresponding matrices.
        """
        ls = []
        rs = []
        df = self.rotations
        for _, row in df.iterrows():
            if row["rotation"]:
                ang_incr = torch.Tensor([360. / row["folds"]]).double()
                for fold in range(1, row["folds"]):
                    angle = torch.Tensor([ang_incr]).double() * fold
                    l = torch.eye(4, dtype=torch.double)
                    ls.append(l)
                    r = self.rotation_3d_matrix_arbitrary(
                        angle, torch.Tensor([row["x"], row["y"], row["z"]]).double()).T
                    r = (abs(r) >= 1e-6) * r
                    rs.append(r)
            elif row["inversion"]:
                l = torch.eye(4, dtype=torch.double)
                l[2, 2] = -1
                ls.append(l)
                r = -torch.eye(4, dtype=torch.double)
                r[3, 3] = 1
                rs.append(r)
            elif row["mirror_plane"]:
                l = torch.eye(4, dtype=torch.double)
                ls.append(l)
                r = torch.eye(4, dtype=torch.double)
                r[2, 2] = -1
                a = self.align_with_z(torch.Tensor([row["x"], row["y"], row["z"]]).double()).T
                r = a @ r @ a.inverse()
                rs.append(r)

        self.ll = torch.stack(ls, dim=0)
        self.rr = torch.stack(rs, dim=0)
        self.compute_subgroup()

    def compute_subgroup(self):
        """
        Expand the set of symmetry matrices by combining existing ones until closure is reached.

        Ensures that all possible products of symmetry operations are included.
        """
        found_r = [None]
        identity = torch.eye(3, dtype=torch.double)
        while len(found_r) != 0:
            found_r = []
            found_l = []
            for r1, l1 in zip(self.rr, self.ll):
                for r2, l2 in zip(self.rr, self.ll):
                    rn = r1 @ r2
                    rn = (abs(rn) >= 1e-6) * rn
                    ln = l1 @ l2
                    ln = (abs(ln) >= 1e-6) * ln
                    tried = False
                    if not torch.allclose(rn[:3, :3], identity) or not torch.allclose(ln[:3, :3], identity):
                        for rt, lt in zip(self.rr, self.ll):
                            if torch.allclose(rn, rt) and torch.allclose(ln, lt):
                                tried = True
                        for rt, lt in zip(found_r, found_l):
                            if torch.allclose(rn, rt) and torch.allclose(ln, lt):
                                tried = True
                        if not tried:
                            found_r.append(rn.unsqueeze(0))
                            found_l.append(ln.unsqueeze(0))

            self.rr = torch.concat([self.rr] + found_r, dim=0)
            self.ll = torch.concat([self.ll] + found_l, dim=0)

    def align_with_z(self, axis: torch.Tensor, homogeneous: bool = True) -> torch.Tensor:
        """
        Create a transformation matrix that aligns the given axis with the Z axis.

        Args:
            axis (torch.Tensor): 3D vector to align with Z.
            homogeneous (bool, optional): If True, returns a 4x4 homogeneous matrix. Defaults to True.

        Returns:
            torch.Tensor: Transformation matrix aligning axis with Z.
        """
        if axis.numel() != 3:
            raise ValueError("Axis must be a 3D vector")
        axis = axis / torch.norm(axis)  # Normalize the axis

        proj_mod = torch.sqrt(axis[1] ** 2 + axis[2] ** 2)
        if proj_mod > torch.finfo(torch.double).eps:  # 1e-6
            # Build a transformation matrix aligning the axis with Z
            result = torch.tensor([
                [proj_mod, -axis[0] * axis[1] / proj_mod, -axis[0] * axis[2] / proj_mod],
                [0, axis[2] / proj_mod, -axis[1] / proj_mod],
                [axis[0], axis[1], axis[2]]
            ], dtype=torch.double)
        else:
            # Axis is aligned with X, handle special case
            result = torch.tensor([
                [0, 0, -1 if axis[0] > 0 else 1],
                [0, 1, 0],
                [1 if axis[0] > 0 else -1, 0, 0]
            ], dtype=torch.double)

        if homogeneous:
            result_h = torch.eye(4, dtype=torch.double)
            result_h[:3, :3] = result
            return result_h
        return result

    def rotation_3d_matrix(self, angle: torch.Tensor, axis: str = 'Z', homogeneous: bool = True) -> torch.Tensor:
        """
        Generate a rotation matrix for a given axis and angle.

        Args:
            angle (torch.Tensor): Rotation angle in degrees.
            axis (str, optional): Axis of rotation ('X', 'Y', or 'Z'). Defaults to 'Z'.
            homogeneous (bool, optional): If True, returns a 4x4 homogeneous matrix. Defaults to True.

        Returns:
            torch.Tensor: Rotation matrix.
        """
        angle = torch.deg2rad(angle)
        cosine, sine = torch.cos(angle), torch.sin(angle)

        if axis == 'Z':
            result = torch.tensor([
                [cosine, -sine, 0, 0] if homogeneous else [cosine, -sine, 0],
                [sine, cosine, 0, 0] if homogeneous else [sine, cosine, 0],
                [0, 0, 1, 0] if homogeneous else [0, 0, 1],
                [0, 0, 0, 1] if homogeneous else []
            ], dtype=torch.double)
        elif axis == 'Y':
            result = torch.tensor([
                [cosine, 0, sine, 0] if homogeneous else [cosine, 0, sine],
                [0, 1, 0, 0] if homogeneous else [0, 1, 0],
                [-sine, 0, cosine, 0] if homogeneous else [-sine, 0, cosine],
                [0, 0, 0, 1] if homogeneous else []
            ], dtype=torch.double)
        elif axis == 'X':
            result = torch.tensor([
                [1, 0, 0, 0] if homogeneous else [1, 0, 0],
                [0, cosine, -sine, 0] if homogeneous else [0, cosine, -sine],
                [0, sine, cosine, 0] if homogeneous else [0, sine, cosine],
                [0, 0, 0, 1] if homogeneous else []
            ], dtype=torch.double)
        else:
            raise ValueError("Unknown axis: choose 'X', 'Y', or 'Z'")

        return result[:3, :3] if not homogeneous else result

    def rotation_3d_matrix_arbitrary(self, angle: torch.Tensor, axis: torch.Tensor, homogeneous=True) -> torch.Tensor:
        """
        Generate a rotation matrix for an arbitrary axis and angle.

        Args:
            angle (torch.Tensor): Rotation angle in degrees.
            axis (torch.Tensor): 3D vector representing the axis of rotation.
            homogeneous (bool, optional): If True, returns a 4x4 homogeneous matrix. Defaults to True.

        Returns:
            torch.Tensor: Rotation matrix for the arbitrary axis.
        """
        axis_rotation = self.align_with_z(axis, homogeneous)
        z_rotation = self.rotation_3d_matrix(angle, 'Z', homogeneous)
        result = axis_rotation.T @ z_rotation @ axis_rotation
        return result
