<div align="center">

<h1>
GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation
</h1>

<div>
    <a href='https://nirvanalan.github.io/' target='_blank'>Yushi Lan</a><sup>1</sup>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Shangchen Zhou</a><sup>1</sup>&emsp;
    <a href='https://zhaoyanglyu.github.io/' target='_blank'>Zhaoyang Lyu</a><sup>1</sup>&emsp;
    <a href='https://hongfz16.github.io' target='_blank'>Fangzhou Hong</a><sup>1</sup>&emsp;
    <a href='https://williamyang1991.github.io/' target='_blank'>Shuai Yang</a><sup>2</sup>&emsp;
    <br>
    <a href='https://daibo.info/' target='_blank'>Bo Dai</a>
    <sup>3</sup>
    <a href='https://xingangpan.github.io/' target='_blank'>Xingang Pan</a>
    <sup>1</sup>
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy</a>
    <sup>1</sup> &emsp;
</div>
<div>
    S-Lab, Nanyang Technological University<sup>1</sup>;
    <!-- &emsp; -->
    <br>
    Wangxuan Institute of Computer Technology, Peking University<sup>2</sup>;
    <br>
    <!-- &emsp; -->
    Shanghai Artificial Intelligence Laboratory <sup>3</sup>
    <!-- <br>
     <sup>*</sup>corresponding author -->
</div>

<div>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNIRVANALAN%2FGaussianAnything&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>
<br>
<!-- <h4> -->
<strong>
GaussianAnything generates <i>high-quality</i> and <i>editable</i> surfel Gaussians through a cascaded native 3D diffusion pipeline, given single-view images or texts as the conditions.
</strong>

<!-- <table>
<tr></tr>
<tr>
    <td>
        <img src="assets/t23d/dit-l2/the-eiffel-tower.gif">
    </td>
    <td>
        <img src="assets/t23d/dit-l2/stone-waterfall-with-wooden-shed.gif">
    </td>
    <td>
        <img src="assets/t23d/dit-l2/a-plate-of-sushi.gif">
    </td>
    <td>
        <img src="assets/t23d/dit-l2/wooden-chest-with-golden-trim.gif">
    </td>
    <td>
        <img src="assets/t23d/dit-l2/a-blue-plastic-chair.gif">
    </td>
</tr>


<tr>
    <td align='center' width='20%'>The eiffel tower.</td>
    <td align='center' width='20%'>A stone waterfall with wooden shed.</td>
    <td align='center' width='20%'>A plate of sushi</td>
    <td align='center' width='20%'>A wooden chest with golden trim</td>
    <td align='center' width='20%'>A blue plastic chair.</td>
</tr>
<tr></tr>
</table> -->

<video width="100%" controls="" loop="" autoplay="" muted="">
            <source src="https://nirvanalan.github.io/projects/GA/static/videos/gallary_video_final.mp4" type="video/mp4">
            Your browser does not support the video tag.
</video>



<!-- <br> -->

For more visual results, go checkout our <a href="https://nirvanalan.github.io/projects/GA/" target="_blank">project page</a> :page_with_curl:

<!-- <strike> -->
Codes coming soon :facepunch:
<!-- </strike> -->

This repository contains the official implementation of GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation

</div>

---

<h4 align="center">
  <a href="https://nirvanalan.github.io/projects/GA/" target='_blank'>[Project Page]</a>
  <!-- •
  <a href="https://arxiv.org/pdf/TODO" target='_blank'>[arXiv]</a>  -->
  <!-- •
  <a href="https://huggingface.co/spaces/yslan/LN3Diff_I23D" target='_blank'>[Gradio Demo]</a>  -->
</h4>


## Abstract

<p> While 3D content generation has advanced significantly, existing methods still face challenges with
            input formats, latent space design, and output representations. This paper introduces a novel 3D
            generation framework that addresses these challenges, offering scalable, high-quality 3D generation
            with an interactive <i>Point Cloud-structured Latent</i> space. Our framework employs a
            Variational Autoencoder (VAE) with multi-view posed RGB-D(epth)-N(ormal) renderings as input, using
            a unique latent space design that preserves 3D shape information, and incorporates a cascaded latent
            diffusion model for improved shape-texture disentanglement. The proposed method, GaussianAnything,
            supports multi-modal conditional 3D generation, allowing for point cloud, caption, and
            single/multi-view image inputs. Notably, the newly proposed latent space naturally enables
            geometry-texture disentanglement, thus allowing 3D-aware editing. Experimental results demonstrate
            the effectiveness of our approach on multiple datasets, outperforming existing methods in both text-
            and image-conditioned 3D generation.</p>

<img class="summary-img" src="https://nirvanalan.github.io/projects/GA/static/images/ga-teaser.jpg" style="width:100%;">


## :mega: Updates

[11/2024] Initial release.


<!-- ### Demo
<img src="./assets/huggingface-screenshot.png"
            alt="Demo screenshot."/>
Check out our online demo on [Gradio space](https://huggingface.co/spaces/yslan/LN3Diff_I23D). To run the demo locally, simply follow the installation instructions below, and afterwards call 

```bash 
bash shell_scripts/final_release/inference/gradio_sample_obajverse_i23d_dit.sh
``` -->


### :dromedary_camel: TODO

- [ ] Release inference code and checkpoints.
- [ ] Release Training code.
- [ ] Release Gradio Demo.




## :handshake: BibTex
If you find our work useful for your research, please consider citing the paper:
```
@article{lan2024ga,
    title={GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation}, 
    author={Yushi Lan and Shangchen Zhou and Zhaoyang Lyu and Fangzhou Hong and Shuai Yang and Bo Dai and Xingang Pan and Chen Change Loy},
    year={2024},
    booktitle={arXiv preprint arxiv},
}
```


## :newspaper_roll:  License

Distributed under the NTU S-Lab License. See `LICENSE` for more information.


## Contact

If you have any question, please feel free to contact us via `lanyushi15@gmail.com` or Github issues.
