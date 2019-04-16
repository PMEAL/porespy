---
title: 'PoreSpy: A Python Toolkit for Quantitative Analysis of Porous Media Images'

tags:
  - Python
  - porous media
  - tomography
  - image analysis

authors:
  - name: Jeff T. Gostick
    orcid: 0000-0001-7736-7124
    affiliation: 1
  - name: Zohaib A. Khan
    orcid: 0000-0003-2115-7798
    affiliation: 1
  - name: Thomas G. Tranter
    orcid: 0000-0003-4721-5941
    affiliation: "1, 2"
  - name: Matthew D.R. Kok
    orcid: 0000-0001-8410-9748
    affiliation: "2, 3"
  - name: Mehrez Agnaou
    orcid: 0000-0002-6635-080X
    affiliation: 1
  - name: Mohammadamin Sadeghi
    orcid: 0000-0002-6756-9117
    affiliation: 3
  - name: Rhodri Jervis
    orcid: 0000-0003-2784-7802
    affiliation: 2

affiliations:
 - name: Department of Chemical Engineering, University of Waterloo, Waterloo, ON, Canada
   index: 1
 - name: Department of Chemical Engineering, University College London, London, United Kingdom
   index: 2
 - name: Department of Chemical Engineering, McGill University, Montreal, QC, Canada
   index: 3

date: 14 April 2019

bibliography: paper.bib

---

# Summary

Porous materials play a central role in many technologies, from micron-thick electrodes used in batteries and fuel-cells [@karanPEFCCatalystLayer2017] to kilometer-long geologic formations of interest in oil recovery, contaminant transport and CO2 sequestration [@kumarReservoirSimulationCO22005].  These applications share a common interest in analyzing the transport processes occurring at the pore-scale, since these ultimately control the observable macroscopic behavior [@anovitzCharacterizationAnalysisPorosity2015].  Images of porous materials are a valuable source of information, since performance and structure are intimately linked through the topology and geometry of the media.  A variety of techniques are available for imaging a material's internal pore structure with sub-micron resolution, including X-ray tomography [@colesPoreLevelImaging1998], ptychography [@dierolfPtychographicXrayComputed2010], and FIB-SEM sectioning [@heaneyFocusedIonBeam2001].  Each of these tools can provide exquisitely detailed images, and retrieving quantitative information from these images has become a vital tool in all areas of porous media research.

The analysis of images, both 2D sections and 3D volumetric data [@taiwoComparisonThreeDimensional2016], is an established field, dating to the advent of computers with sufficient memory to store and manipulate the rather large file sizes.  Pioneering work by Torquato, summarized in his textbook [@torquatoRandomHeterogeneousMaterials2005], laid a solid foundation by defining many metrics still widely used, including chord-length distribution [@jervisXrayNanoComputed2018] and radial-density function [@kokInsightsEffectStructural2018].  The field is far from dormant, however, as increasingly powerful computers allow application of concomitantly advanced algorithms, such as watershed partitioning to create networks [@khanDualNetworkExtraction2019, @gostickVersatileEfficientPore2017] for pore-network modeling [@gostickOpenPNMPoreNetwork2016], and direct numerical simulations [@garcia-salaberriImplicationsInherentInhomogeneities2019] which can often supplant experiments [@garcia-salaberriAnalysisRepresentativeElementary2018] or even conduct experiments that are not physically possible [@garcia-salaberriEffectiveDiffusivityPartiallysaturated2015, @kokMassTransferFibrous2019].  The aim of PoreSpy is to collect the various tools and functions needed by researchers to perform quantitative analysis of volumetric images of porous materials into a single, easy to use, open-source package.  PoreSpy is written in the popular Python language which contains several image processing libraries offering the full suite of necessary tools [@gouillartAnalyzingMicrotomographyData2016, @scikit-image].  PoreSpy contains a large, though not exhaustive, set of the most commonly used tools for porous media analysis, which typically work by applying a series of the basic image analysis functions with appropriate arguments and adjustments. The functions in PoreSpy are organized into the following categories:

- **generators**: *create artificial images*
- **filters**: *highlight features of interest*
- **metrics**: *extract quantitative information*
- **networks**: *pore-network representations*
- **tools**: *helper functions*
- **io**: *convert between formats*
- **visualization**: *create basic views*

The scope of PoreSpy is constrained to tools for the analysis and manipulation of images.  As such it does not attempt to offer 3D visualization which is already available in open-source software such as Paraview [@ayachitParaViewGuideParallel2015] and numerous sophisticated commercial packages (e.g. DragonFly, Avizo).  PoreSpy also stops short, at least for now, of performing direct numerical simulations on images, despite this being a common and powerful use of volumetric images, since such algorithms are extremely complex and best handled by dedicated packages.  Instead, PoreSpy offers tools for converting images into file formats suitable for these other packages.

Being implemented in Python means it can be installed with a single command (``pip install porespy``) without any compilation, and is accessible to researchers with any level of programming experience via Jupyter notebooks.  The design of PoreSpy is modelled after other related packages in the Scipy stack, as a collection of simple functions with well-defined and clearly documented inputs and outputs.   It is open-source so that other researchers can confidently use each function knowing how it works, but also so that other researchers can contribute new algorithms or Python-implementations of seminal tools.


# Acknowledgements

MDRK acknowledges support from EPSRC under grant EP/R020973/1.  MDRK and RJ acknowledge the support of the Faraday Institution Degradation and Multi scale Modelling projects.  ZA express his appreciation to the University of Engineering and Technology Lahore, Pakistan for their funding and support.

# References
