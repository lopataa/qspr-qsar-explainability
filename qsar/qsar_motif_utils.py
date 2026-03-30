from pathlib import Path
import io
import pickle

import numpy as np
from PIL import Image


def fingerprint_mol_with_bit_info(mol, radius, n_bits, generator=None):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if generator is None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()

    fp = generator.GetFingerprint(mol, additionalOutput=additional_output)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    raw_bit_info = additional_output.GetBitInfoMap()
    bit_info_map = {
        int(bit): [(int(atom_idx), int(env_radius)) for atom_idx, env_radius in atom_info]
        for bit, atom_info in raw_bit_info.items()
    }
    return arr, bit_info_map


def _morgan_environment_atoms_and_bonds(mol, center_atom_idx, radius):
    from rdkit import Chem

    bond_ids = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom_idx))
    atom_ids = {center_atom_idx}
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(bond_id)
        atom_ids.add(bond.GetBeginAtomIdx())
        atom_ids.add(bond.GetEndAtomIdx())
    return sorted(atom_ids), sorted(bond_ids)


def rank_active_morgan_bits(bit_info_map, bit_scores, top_n):
    ranked = []
    for bit, occurrences in bit_info_map.items():
        score = float(bit_scores[bit])
        if score <= 0:
            continue
        ranked.append((int(bit), score, len(occurrences)))
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked[:top_n]


def _normalize_scores(scores):
    if len(scores) == 0:
        return np.asarray([], dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return np.ones_like(values)
    return (values - low) / (high - low)


def _lerp_rgb(color_a, color_b, t):
    return tuple(float(color_a[i] + (color_b[i] - color_a[i]) * t) for i in range(3))


def _prepare_morgan_bit_render_payload(
    mol,
    bit_info_map,
    bit_scores,
    top_n,
    min_alpha,
    max_alpha,
    low_color,
    high_color,
):
    from rdkit import Chem

    ranked_bits = rank_active_morgan_bits(bit_info_map, bit_scores, top_n=top_n)
    if not ranked_bits:
        raise ValueError("No active bits with positive score were found for this molecule.")
    if not (0.0 <= min_alpha <= max_alpha < 1.0):
        raise ValueError("Expected alpha range to satisfy 0 <= min_alpha <= max_alpha < 1.")

    score_values = [score for _, score, _ in ranked_bits]
    norm_scores = _normalize_scores(score_values)

    render_items = []
    bit_summaries = []

    for item_idx, (bit, score, occurrences_count) in enumerate(ranked_bits):
        rank = item_idx + 1
        highlighted_atoms = set()
        highlighted_bonds = set()
        fragments = set()

        for center_atom_idx, radius in bit_info_map[bit]:
            atom_ids, bond_ids = _morgan_environment_atoms_and_bonds(mol, center_atom_idx, radius)
            highlighted_atoms.update(atom_ids)
            highlighted_bonds.update(bond_ids)

            fragment = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=atom_ids,
                bondsToUse=bond_ids,
                canonical=True,
                isomericSmiles=False,
            )
            if fragment:
                fragments.add(fragment)

        norm_score = float(norm_scores[item_idx])
        alpha = float(min_alpha + (max_alpha - min_alpha) * norm_score)
        rgb = _lerp_rgb(low_color, high_color, norm_score)
        rgba = (*rgb, alpha)

        render_items.append(
            {
                "legend": f"#{rank} bit {bit}\nimportance={score:.5f} (norm={norm_score:.2f})",
                "atoms": sorted(highlighted_atoms),
                "bonds": sorted(highlighted_bonds),
                "rgba": rgba,
            }
        )
        bit_summaries.append(
            {
                "rank": rank,
                "bit": bit,
                "importance": score,
                "importance_norm": norm_score,
                "highlight_alpha": alpha,
                "occurrences": occurrences_count,
                "fragments": "; ".join(sorted(fragments)),
            }
        )

    return render_items, bit_summaries


def _composite_rgba(values, min_alpha):
    if len(values) == 1:
        return values[0]

    out_prgb = np.zeros(3, dtype=np.float32)
    out_alpha = 0.0

    for rgba in reversed(values):
        src_rgb = np.asarray(rgba[:3], dtype=np.float32)
        src_alpha = float(rgba[3])
        src_prgb = src_rgb * src_alpha
        out_prgb = src_prgb + out_prgb * (1.0 - src_alpha)
        out_alpha = src_alpha + out_alpha * (1.0 - src_alpha)

    if np.isclose(out_alpha, 0.0):
        rgb = (0.0, 0.0, 0.0)
    else:
        rgb = tuple(float(channel) for channel in (out_prgb / out_alpha))

    alpha_values = np.asarray([val[3] for val in values], dtype=np.float32)
    alpha = float(np.clip(alpha_values.max(), min_alpha, 0.85))
    return (*rgb, alpha)


def draw_morgan_bit_grid(
    mol,
    bit_info_map,
    bit_scores,
    top_n=6,
    mols_per_row=3,
    sub_img_size=(420, 320),
    min_alpha=0.0,
    max_alpha=0.75,
    low_color=(0.62, 0.86, 0.98),
    high_color=(0.60, 0.46, 0.95),
    use_svg=False,
):
    from rdkit.Chem import Draw

    render_items, bit_summaries = _prepare_morgan_bit_render_payload(
        mol=mol,
        bit_info_map=bit_info_map,
        bit_scores=bit_scores,
        top_n=top_n,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        low_color=low_color,
        high_color=high_color,
    )

    mols = [mol] * len(render_items)
    legends = [item["legend"] for item in render_items]
    atom_highlights = [item["atoms"] for item in render_items]
    bond_highlights = [item["bonds"] for item in render_items]
    atom_colors = [{atom_idx: item["rgba"] for atom_idx in item["atoms"]} for item in render_items]
    bond_colors = [{bond_idx: item["rgba"] for bond_idx in item["bonds"]} for item in render_items]

    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        highlightAtomLists=atom_highlights,
        highlightBondLists=bond_highlights,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        useSVG=bool(use_svg),
    )
    return image, bit_summaries


def draw_morgan_bit_overlay(
    mol,
    bit_info_map,
    bit_scores,
    top_n=6,
    mols_per_row=3,
    sub_img_size=(420, 320),
    min_alpha=0.0,
    max_alpha=0.75,
    low_color=(0.62, 0.86, 0.98),
    high_color=(0.60, 0.46, 0.95),
    use_svg=False,
):
    from rdkit.Chem import Draw

    render_items, bit_summaries = _prepare_morgan_bit_render_payload(
        mol=mol,
        bit_info_map=bit_info_map,
        bit_scores=bit_scores,
        top_n=top_n,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        low_color=low_color,
        high_color=high_color,
    )

    atom_contribs = {}
    bond_contribs = {}
    for item in render_items:
        for atom_idx in item["atoms"]:
            atom_contribs.setdefault(atom_idx, []).append(item["rgba"])
        for bond_idx in item["bonds"]:
            bond_contribs.setdefault(bond_idx, []).append(item["rgba"])

    atom_colors = {
        atom_idx: _composite_rgba(values, min_alpha=min_alpha)
        for atom_idx, values in atom_contribs.items()
    }
    bond_colors = {
        bond_idx: _composite_rgba(values, min_alpha=min_alpha)
        for bond_idx, values in bond_contribs.items()
    }
    size = (
        int(sub_img_size[0] * max(1, mols_per_row)),
        int(sub_img_size[1] * 1.15),
    )
    image = Draw.MolsToGridImage(
        [mol],
        molsPerRow=1,
        subImgSize=size,
        highlightAtomLists=[sorted(atom_contribs)],
        highlightBondLists=[sorted(bond_contribs)],
        highlightAtomColors=[atom_colors],
        highlightBondColors=[bond_colors],
        useSVG=bool(use_svg),
    )
    return image, bit_summaries


def compute_atom_weights_from_top_bits(
    mol,
    bit_info_map,
    bit_scores,
    top_n,
    aggregation="sum",
):
    if aggregation not in {"sum", "mean"}:
        raise ValueError("aggregation must be either 'sum' or 'mean'.")

    ranked_bits = rank_active_morgan_bits(bit_info_map, bit_scores, top_n=top_n)
    if not ranked_bits:
        raise ValueError("No active bits with positive score were found for this molecule.")

    n_atoms = int(mol.GetNumAtoms())
    atom_weights = np.zeros((n_atoms,), dtype=np.float32)
    atom_counts = np.zeros((n_atoms,), dtype=np.int32)

    for bit, score, _ in ranked_bits:
        score_value = float(score)
        for center_atom_idx, radius in bit_info_map.get(int(bit), []):
            atom_ids, _ = _morgan_environment_atoms_and_bonds(mol, center_atom_idx, radius)
            for atom_idx in atom_ids:
                atom_weights[int(atom_idx)] += score_value
                atom_counts[int(atom_idx)] += 1

    if aggregation == "mean":
        mask = atom_counts > 0
        atom_weights[mask] = atom_weights[mask] / atom_counts[mask]

    return atom_weights


def render_similarity_map_image(mol, atom_weights, draw_size=700):
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import SimilarityMaps

    drawer = Draw.MolDraw2DCairo(int(draw_size), int(draw_size))
    SimilarityMaps.GetSimilarityMapFromWeights(mol, list(map(float, atom_weights)), draw2d=drawer)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def save_similarity_map_png(mol, atom_weights, output_path, draw_size=700):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_similarity_map_image(mol, atom_weights, draw_size=draw_size)
    image.save(output_path)
    return output_path


def save_similarity_map_svg(mol, atom_weights, output_path, draw_size=700):
    from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drawer = rdMolDraw2D.MolDraw2DSVG(int(draw_size), int(draw_size))
    SimilarityMaps.GetSimilarityMapFromWeights(mol, list(map(float, atom_weights)), draw2d=drawer)
    drawer.FinishDrawing()
    output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")
    return output_path


def render_plain_molecule_image(mol, draw_size=700):
    from rdkit.Chem import Draw

    image = Draw.MolToImage(mol, size=(int(draw_size), int(draw_size)))
    return image.convert("RGBA")


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_pickle_cache(cache_path, expected_meta):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("meta") != expected_meta:
        return None
    return payload.get("data")


def save_pickle_cache(cache_path, meta, data):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "data": data}
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
