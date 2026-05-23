# (IEEE TMI) Refine Then Fusion: Robust 3D Brain MRI Synthesis via Vision–Language Collaboration



<div align="center">

[**Jinbao Wei**](https://scholar.google.com/citations?user=KcKJI6MAAAAJ&hl=zh-CN),
[**Gang Yang**](https://scholar.google.com/citations?user=gctrxXsAAAAJ&hl=zh-CN&oi=ao),
[**Wei Wei**](https://orcid.org/0000-0003-4097-1592),
[**Aiping Liu**](https://scholar.google.com/citations?user=vEDf62sAAAAJ&hl=zh-CN),
[**Xun Chen**](https://scholar.google.com/citations?user=aBnUWyQAAAAJ&hl=zh-CN),

</div>



[[Paper]](https://ieeexplore.ieee.org/abstract/document/11516481)


## Introduction
Metadata-guided cross-modality 3D MRI synthesis aims to generate target-contrast volumes from source-modality data conditioned on clinically available metadata, which is important for enhancing clinical imaging flexibility. However, existing methods still suffer from two main limitations: 1) They neglect spatial dependencies within volumetric representations, yielding structurally ambiguous features that blur anatomical boundaries and hinder precise semantic integration. 2) They rely on conventional cross-attention between visual and textual features, limiting the precision of visual-semantic alignment, which reduces robustness across challenging conditions. To address these issues, we propose RTFSyn, a metadata-guided 3D MRI synthesis framework that achieves effective vision–language collaboration through a refine-then-fusion paradigm. The proposed RTFSyn benefits from several merits. First, we design an axis-aware visual refinement module that captures directional dependencies within volumetric features, enabling redundancy suppression and improved structural representation before fusion. Second, we propose a cross-modal adaptive fusion module that leverages pixel packing–recovery to realize efficient cross-attention for improved alignment, while text-conditioned dynamic convolution enables fine-grained semantic injection, together enhancing vision–language collaboration. Lastly, an implicit neural decoder reconstructs the target modality as a continuous function, enabling flexible high-fidelity synthesis. Under this synergistic paradigm, RTFSyn seamlessly unites robust spatial refinement with adaptive feature fusion to achieve highly precise cross-modal alignment. Extensive experiments across four multi-center datasets demonstrate that RTFSyn not only surpasses state-of-the-art methods quantitatively, but also exhibits robust performance under diverse imaging artifacts, zero-shot evaluations, and multi-dimensional clinical validations, all with favorable computational efficiency. The high fidelity, robustness, and efficiency of RTFSyn demonstrate its great potential for clinical applications.






## Citations

You may want to cite:
```
@article{wei2026refine,
  title={Refine Then Fusion: Robust 3D Brain MRI Synthesis via Vision--Language Collaboration},
  author={Wei, Jinbao and Yang, Gang and Wei, Wei and Liu, Aiping and Chen, Xun},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}

```
