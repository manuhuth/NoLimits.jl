import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import { juliaReplTransformer } from './julia-repl-transformer'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/NoLimits.jl/dev/',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: [
{ text: 'Home', link: '/index' },
{ text: 'Getting Started', collapsed: false, items: [
{ text: 'Installation', link: '/installation' },
{ text: 'Quickstart', link: '/quickstart' },
{ text: 'Capabilities', link: '/capabilities' },
{ text: 'NLME Methodology', link: '/nlme-methodology' }]
 },
{ text: 'Model Building', collapsed: false, items: [
{ text: 'Overview', link: '/model-building/index' },
{ text: '@Model', link: '/model-building/model-macro' },
{ text: '@helpers', link: '/model-building/helpers' },
{ text: '@fixedEffects', link: '/model-building/fixed-effects' },
{ text: '@covariates', link: '/model-building/covariates' },
{ text: '@randomEffects', link: '/model-building/random-effects' },
{ text: '@preDifferentialEquation', link: '/model-building/pre-differential-equation' },
{ text: '@DifferentialEquation', link: '/model-building/differential-equation' },
{ text: '@initialDE', link: '/model-building/initial-de' },
{ text: '@formulas', link: '/model-building/formulas' },
{ text: 'Function Approximators (NNs + SoftTrees)', link: '/model-building/universal-function-approximators' }]
 },
{ text: 'Data Model Construction', link: '/data-model-construction' },
{ text: 'Estimation', collapsed: false, items: [
{ text: 'Overview', link: '/estimation/index' },
{ text: 'MLE / MAP', link: '/estimation/mle' },
{ text: 'Laplace / LaplaceMAP', link: '/estimation/laplace' },
{ text: 'FOCEI / FOCEIMAP', link: '/estimation/focei' },
{ text: 'GH Quadrature', link: '/estimation/ghquadrature' },
{ text: 'MCEM', link: '/estimation/mcem' },
{ text: 'SAEM', link: '/estimation/saem' },
{ text: 'Pooled / PooledMap', link: '/estimation/pooled' },
{ text: 'MCMC', link: '/estimation/mcmc' },
{ text: 'VI', link: '/estimation/vi' },
{ text: 'Multistart', link: '/estimation/multistart' },
{ text: 'Cross-Validation', link: '/estimation/cv' },
{ text: 'Saving & Loading', link: '/estimation/saving-and-loading' }]
 },
{ text: 'Uncertainty Quantification', collapsed: false, items: [
{ text: 'Overview', link: '/uncertainty-quantification/index' },
{ text: 'Wald', link: '/uncertainty-quantification/wald' },
{ text: 'Profile likelihood', link: '/uncertainty-quantification/profile-likelihood' },
{ text: 'MCMC-based uncertainty', link: '/uncertainty-quantification/mcmc-based-uncertainty' }]
 },
{ text: 'Plotting', link: '/plotting/index' },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Multi-Method Comparison', link: '/tutorials/mixed-effects-multiple-methods' },
{ text: 'ODE Model with Dosing (MCEM)', link: '/tutorials/mixed-effects-ode-mcem' },
{ text: 'Neural Differential Equations (SAEM)', link: '/tutorials/mixed-effects-nn-saem' },
{ text: 'Soft-Tree Differential Equations (SAEM)', link: '/tutorials/mixed-effects-softtree-saem' },
{ text: 'Count Outcomes: Poisson & NegativeBinomial (MCEM)', link: '/tutorials/mixed-effects-seizure-counts-poisson-nb-mcem' },
{ text: 'Left-Censored Nonlinear Model (Laplace)', link: '/tutorials/mixed-effects-left-censored-virload50-laplace' },
{ text: 'Fixed-Effects: MLE & MAP', link: '/tutorials/fixed-effects-nonlinear-mle-map' },
{ text: 'Fixed-Effects: Variational Inference', link: '/tutorials/fixed-effects-vi' },
{ text: 'Hidden & Observed Markov Models', link: '/tutorials/markov-models-observed-hidden-coarsed' }]
 },
{ text: 'Developers Guide', link: '/developers-guide' },
{ text: 'How to Contribute', link: '/how-to-contribute' },
{ text: 'References', link: '/references' },
{ text: 'API', link: '/api' }
]
,
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/NoLimits.jl/dev/',// TODO: replace this in makedocs!
  title: 'NoLimits.jl',
  description: 'Documentation for NoLimits.jl',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [
    
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  
  markdown: {
    codeTransformers: [juliaReplTransformer()],
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/NoLimits.jl'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: { src: '/logo.png', width: 24, height: 24},
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
{ text: 'Home', link: '/index' },
{ text: 'Getting Started', collapsed: false, items: [
{ text: 'Installation', link: '/installation' },
{ text: 'Quickstart', link: '/quickstart' },
{ text: 'Capabilities', link: '/capabilities' },
{ text: 'NLME Methodology', link: '/nlme-methodology' }]
 },
{ text: 'Model Building', collapsed: false, items: [
{ text: 'Overview', link: '/model-building/index' },
{ text: '@Model', link: '/model-building/model-macro' },
{ text: '@helpers', link: '/model-building/helpers' },
{ text: '@fixedEffects', link: '/model-building/fixed-effects' },
{ text: '@covariates', link: '/model-building/covariates' },
{ text: '@randomEffects', link: '/model-building/random-effects' },
{ text: '@preDifferentialEquation', link: '/model-building/pre-differential-equation' },
{ text: '@DifferentialEquation', link: '/model-building/differential-equation' },
{ text: '@initialDE', link: '/model-building/initial-de' },
{ text: '@formulas', link: '/model-building/formulas' },
{ text: 'Function Approximators (NNs + SoftTrees)', link: '/model-building/universal-function-approximators' }]
 },
{ text: 'Data Model Construction', link: '/data-model-construction' },
{ text: 'Estimation', collapsed: false, items: [
{ text: 'Overview', link: '/estimation/index' },
{ text: 'MLE / MAP', link: '/estimation/mle' },
{ text: 'Laplace / LaplaceMAP', link: '/estimation/laplace' },
{ text: 'FOCEI / FOCEIMAP', link: '/estimation/focei' },
{ text: 'GH Quadrature', link: '/estimation/ghquadrature' },
{ text: 'MCEM', link: '/estimation/mcem' },
{ text: 'SAEM', link: '/estimation/saem' },
{ text: 'Pooled / PooledMap', link: '/estimation/pooled' },
{ text: 'MCMC', link: '/estimation/mcmc' },
{ text: 'VI', link: '/estimation/vi' },
{ text: 'Multistart', link: '/estimation/multistart' },
{ text: 'Cross-Validation', link: '/estimation/cv' },
{ text: 'Saving & Loading', link: '/estimation/saving-and-loading' }]
 },
{ text: 'Uncertainty Quantification', collapsed: false, items: [
{ text: 'Overview', link: '/uncertainty-quantification/index' },
{ text: 'Wald', link: '/uncertainty-quantification/wald' },
{ text: 'Profile likelihood', link: '/uncertainty-quantification/profile-likelihood' },
{ text: 'MCMC-based uncertainty', link: '/uncertainty-quantification/mcmc-based-uncertainty' }]
 },
{ text: 'Plotting', link: '/plotting/index' },
{ text: 'Tutorials', collapsed: false, items: [
{ text: 'Multi-Method Comparison', link: '/tutorials/mixed-effects-multiple-methods' },
{ text: 'ODE Model with Dosing (MCEM)', link: '/tutorials/mixed-effects-ode-mcem' },
{ text: 'Neural Differential Equations (SAEM)', link: '/tutorials/mixed-effects-nn-saem' },
{ text: 'Soft-Tree Differential Equations (SAEM)', link: '/tutorials/mixed-effects-softtree-saem' },
{ text: 'Count Outcomes: Poisson & NegativeBinomial (MCEM)', link: '/tutorials/mixed-effects-seizure-counts-poisson-nb-mcem' },
{ text: 'Left-Censored Nonlinear Model (Laplace)', link: '/tutorials/mixed-effects-left-censored-virload50-laplace' },
{ text: 'Fixed-Effects: MLE & MAP', link: '/tutorials/fixed-effects-nonlinear-mle-map' },
{ text: 'Fixed-Effects: Variational Inference', link: '/tutorials/fixed-effects-vi' },
{ text: 'Hidden & Observed Markov Models', link: '/tutorials/markov-models-observed-hidden-coarsed' }]
 },
{ text: 'Developers Guide', link: '/developers-guide' },
{ text: 'How to Contribute', link: '/how-to-contribute' },
{ text: 'References', link: '/references' },
{ text: 'API', link: '/api' }
]
,
    sidebarDrawer: false,
    editLink: { pattern: "https://github.com/manuhuth/NoLimits.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/manuhuth/NoLimits.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
