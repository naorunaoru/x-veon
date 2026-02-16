/* OpenDRT v1.0.0 -------------------------------------------------


  Written by Jed Smith
  https://github.com/jedypod/open-display-transform
  License: GPLv3
  -------------------------------------------------*/

/* // Tonescale Parameters */
/* DEFINE_UI_PARAMS(Lp, Lp, DCTLUI_SLIDER_FLOAT, 100.0, 100.0, 1000.0, 0.0) */
/* DEFINE_UI_PARAMS(Lg, Lg, DCTLUI_SLIDER_FLOAT, 10.0, 3.0, 30.0, 0.0) */
/* DEFINE_UI_PARAMS(Lgb, Lg boost, DCTLUI_SLIDER_FLOAT, 0.12, 0.0, 0.5, 0.0) */
/* DEFINE_UI_PARAMS(p, contrast, DCTLUI_SLIDER_FLOAT, 1.4, 1.0, 2.0, 0.0) */
/* DEFINE_UI_PARAMS(toe, toe, DCTLUI_SLIDER_FLOAT, 0.001, 0.0, 0.02, 0.0) */

/* // Color Parameters */
/* DEFINE_UI_PARAMS(pc_p0, purity compress low, DCTLUI_SLIDER_FLOAT, 0.4, 0.0, 1.0, 0.0) */
/* DEFINE_UI_PARAMS(pc_p1, purity compress high, DCTLUI_SLIDER_FLOAT, 0.8, 0.0, 1.0, 0.0) */
/* DEFINE_UI_PARAMS(pb_m0, purity low, DCTLUI_SLIDER_FLOAT, 1.3, 1.0, 2.0, 0.0) */
/* DEFINE_UI_PARAMS(pb_m1, purity high, DCTLUI_SLIDER_FLOAT, 0.5, 0.0, 1.0, 0.0) */
/* DEFINE_UI_PARAMS(base_look, base look, DCTLUI_CHECK_BOX, 0) */

/* // Encoding / IO */
/* DEFINE_UI_PARAMS(in_gamut, in gamut, DCTLUI_COMBO_BOX, 15, {i_xyz, i_ap0, i_ap1, i_p3d65, i_rec2020, i_rec709, i_awg3, i_awg4, i_rwg, i_sgamut3, i_sgamut3cine, i_vgamut, i_bmdwg, i_egamut, i_egamut2, i_davinciwg}, {XYZ, ACES 2065-1, ACEScg, P3D65, Rec.2020, Rec.709, Arri Wide Gamut 3, Arri Wide Gamut 4, Red Wide Gamut RGB, Sony SGamut3, Sony SGamut3Cine, Panasonic V-Gamut, Blackmagic Wide Gamut, Filmlight E-Gamut, Filmlight E-Gamut2, DaVinci Wide Gamut}) */
/* DEFINE_UI_PARAMS(in_oetf, in transfer function, DCTLUI_COMBO_BOX, 0, {ioetf_linear, ioetf_davinci_intermediate, ioetf_filmlight_tlog, ioetf_arri_logc3, ioetf_arri_logc4, ioetf_panasonic_vlog, ioetf_sony_slog3, ioetf_fuji_flog}, {Linear, Davinci Intermediate, Filmlight T-Log, Arri LogC3, Arri LogC4, Panasonic V-Log, Sony S-Log3, Fuji F-Log}) */
/* DEFINE_UI_PARAMS(display_gamut, display gamut, DCTLUI_COMBO_BOX, 0, {Rec709, P3D65, Rec2020}, {Rec.709, P3 D65, Rec.2020}) */
/* DEFINE_UI_PARAMS(EOTF, display eotf, DCTLUI_COMBO_BOX, 2, {lin, srgb, rec1886, dci, pq, hlg}, {Linear, 2.2 Power sRGB Display, 2.4 Power Rec .1886, 2.6 Power DCI, ST 2084 PQ, HLG}) */


float fmin(float a, float b)
{
    if (a < b) {
        return a;
    } else {
        return b;
    }
}


float fmax(float a, float b)
{
    if (a > b) {
        return a;
    } else {
        return b;
    }
}


float ite(bool cond, float t, float e)
{
    if (cond) {
        return t;
    } else {
        return e;
    }
}


const float log2_val = log(2);

float log2(float x)
{
    float y = x;
    if (y < 0) {
        y = 1e-20;
    }
    return log(y) / log2_val;
}

const float SQRT3 = 1.73205080756887729353;
const float PI =  3.14159265358979323846;

struct float3 {
    float x;
    float y;
    float z;
};

struct float3x3 {
    float3 x;
    float3 y;
    float3 z;
};

float3 make_float3(float a, float b, float c)
{
    float3 res = { a, b, c };
    return res;
}

// Helper function to create a float3x3
float3x3 make_float3x3(float3 a, float3 b, float3 c)
{
    float3x3 d;
    d.x = a;
    d.y = b;
    d.z = c;
    return d;
}


float3 mul_f3f(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
float3 add_f3f(float3 a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
float3 add_ff3(float a, float3 b) { return add_f3f(b, a); }
float3 sub_f3f(float3 a, float b) { return make_float3(a.x - b, a.y - b, a.z - b); }
float3 sub_ff3(float a, float3 b) { return make_float3(a - b.x, a - b.y, a - b.z); }
float3 add_f3f3(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

float clampf(float x, float mn, float mx)
{
    return fmin(fmax(x, mn), mx);
}

// Gamut Conversion Matrices
const float3x3 matrix_ap0_to_xyz = make_float3x3(make_float3(0.93863094875, -0.00574192055, 0.017566898852), make_float3(0.338093594922, 0.727213902811, -0.065307497733), make_float3(0.000723121511, 0.000818441849, 1.0875161874));
const float3x3 matrix_ap1_to_xyz = make_float3x3(make_float3(0.652418717672, 0.127179925538, 0.170857283842), make_float3(0.268064059194, 0.672464478993, 0.059471461813), make_float3(-0.00546992851, 0.005182799977, 1.08934487929));
const float3x3 matrix_rec709_to_xyz = make_float3x3(make_float3(0.412390917540, 0.357584357262, 0.180480793118), make_float3(0.212639078498, 0.715168714523, 0.072192311287), make_float3(0.019330825657, 0.119194783270, 0.950532138348));
const float3x3 matrix_p3d65_to_xyz = make_float3x3(make_float3(0.486571133137, 0.265667706728, 0.198217317462), make_float3(0.228974640369, 0.691738605499, 0.079286918044), make_float3(0.0, 0.045113388449, 1.043944478035));
const float3x3 matrix_rec2020_to_xyz = make_float3x3(make_float3(0.636958122253, 0.144616916776, 0.168880969286), make_float3(0.262700229883, 0.677998125553, 0.059301715344), make_float3(0.0, 0.028072696179, 1.060985088348));
const float3x3 matrix_arriwg3_to_xyz = make_float3x3(make_float3(0.638007619284, 0.214703856337, 0.097744451431), make_float3(0.291953779, 0.823841041511, -0.11579482051), make_float3(0.002798279032, -0.067034235689, 1.15329370742));
const float3x3 matrix_arriwg4_to_xyz = make_float3x3(make_float3(0.704858320407, 0.12976029517, 0.115837311474), make_float3(0.254524176404, 0.781477732712, -0.036001909116), make_float3(0.0, 0.0, 1.08905775076));
const float3x3 matrix_redwg_to_xyz = make_float3x3(make_float3(0.735275208950, 0.068609409034, 0.146571278572), make_float3(0.286694079638, 0.842979073524, -0.129673242569), make_float3(-0.079680845141, -0.347343206406, 1.516081929207));
const float3x3 matrix_sonysgamut3_to_xyz = make_float3x3(make_float3(0.706482713192, 0.128801049791, 0.115172164069), make_float3(0.270979670813, 0.786606411221, -0.057586082034), make_float3(-0.009677845386, 0.004600037493, 1.09413555865));
const float3x3 matrix_sonysgamut3cine_to_xyz = make_float3x3(make_float3(0.599083920758, 0.248925516115, 0.102446490178), make_float3(0.215075820116, 0.885068501744, -0.100144321859), make_float3(-0.032065849545, -0.027658390679, 1.14878199098));
const float3x3 matrix_vgamut_to_xyz = make_float3x3(make_float3(0.679644469878, 0.15221141244, 0.118600044733), make_float3(0.26068555009, 0.77489446333, -0.03558001342), make_float3(-0.009310198218, -0.004612467044, 1.10298041602));
const float3x3 matrix_bmdwg_to_xyz = make_float3x3(make_float3(0.606538414955, 0.220412746072, 0.123504832387), make_float3(0.267992943525, 0.832748472691, -0.100741356611), make_float3(-0.029442556202, -0.086612440646, 1.205112814903));
const float3x3 matrix_egamut_to_xyz = make_float3x3(make_float3(0.705396831036, 0.164041340351, 0.081017754972), make_float3(0.280130714178, 0.820206701756, -0.100337378681), make_float3(-0.103781513870, -0.072907261550, 1.265746593475));
const float3x3 matrix_egamut2_to_xyz = make_float3x3(make_float3(0.736477700184, 0.130739651087, 0.083238575781), make_float3(0.275069984406, 0.828017790216, -0.103087774621), make_float3(-0.124225154248, -0.087159767391, 1.3004426724));
const float3x3 matrix_davinciwg_to_xyz = make_float3x3(make_float3(0.700622320175, 0.148774802685, 0.101058728993), make_float3(0.274118483067, 0.873631775379, -0.147750422359), make_float3(-0.098962903023, -0.137895315886, 1.325916051865));
const float3x3 matrix_xyz_to_rec709 = make_float3x3(make_float3(3.2409699419, -1.53738317757, -0.498610760293), make_float3(-0.969243636281, 1.87596750151, 0.041555057407), make_float3(0.055630079697, -0.203976958889, 1.05697151424));
const float3x3 matrix_xyz_to_rec2020 = make_float3x3(make_float3(1.71665118797, -0.355670783776, -0.253366281374), make_float3(-0.666684351832, 1.61648123664, 0.015768545814), make_float3(0.017639857445, -0.042770613258, 0.942103121235));
const float3x3 matrix_xyz_to_p3d65 = make_float3x3(make_float3(2.49349691194, -0.931383617919, -0.402710784451), make_float3(-0.829488969562, 1.76266406032, 0.023624685842), make_float3(0.035845830244, -0.076172389268, 0.956884524008));

// Display gamuts with Normalized adaptation matrices for other creative whitepoints (CAT02)
const float3x3 matrix_p3_to_p3_d50 = make_float3x3(make_float3(0.9287127388, 0.06578032793, 0.005506708345), make_float3(-0.002887159176, 0.8640709228, 4.3593718e-05), make_float3(-0.001009551548, -0.01073503317, 0.6672692039));
const float3x3 matrix_p3_to_p3_d55 = make_float3x3(make_float3(0.9559790976, 0.0403850003, 0.003639287409), make_float3(-0.001771929896, 0.9163058305, 3.3300759e-05), make_float3(-0.000674760809, -0.0072466358, 0.7831189153));
const float3x3 matrix_p3_to_p3_d60 = make_float3x3(make_float3(0.979832881, 0.01836378979, 0.001803284786), make_float3(-0.000805359793, 0.9618000331, 1.8876121e-05), make_float3(-0.000338382322, -0.003671835795, 0.894139105));
const float3x3 matrix_p3_to_rec709_d50 = make_float3x3(make_float3(1.103807322, -0.1103425121, 0.006531676079), make_float3(-0.04079386701, 0.8704694227, -0.000180522628), make_float3(-0.01854055914, -0.07857582481, 0.7105498861));
const float3x3 matrix_p3_to_rec709_d55 = make_float3x3(make_float3(1.149327514, -0.1536910745, 0.004366526746), make_float3(-0.0412590771, 0.9351717477, -0.000116126221), make_float3(-0.01900949528, -0.07928282823, 0.8437884317));
const float3x3 matrix_p3_to_rec709_d60 = make_float3x3(make_float3(1.189986856, -0.192168414, 0.002185496045), make_float3(-0.04168263635, 0.9927757018, -5.5660878e-05), make_float3(-0.01937995127, -0.07933006919, 0.9734397041));
const float3x3 matrix_p3_to_rec709_d65 = make_float3x3(make_float3(1.224940181, -0.2249402404, 0.0), make_float3(-0.04205697775, 1.042057037, -1.4901e-08), make_float3(-0.01963755488, -0.07863604277, 1.098273635));
const float3x3 matrix_p3_to_rec2020 = make_float3x3(make_float3(0.7538330344, 0.1985973691, 0.04756959659), make_float3(0.04574384897, 0.9417772198, 0.01247893122), make_float3(-0.001210340355, 0.0176017173, 0.9836086231));

/* Math helper functions ----------------------------*/

// Return identity 3x3 matrix
float3x3 identity() {
  return make_float3x3(make_float3(1.0, 0.0, 0.0), make_float3(0.0, 1.0, 0.0), make_float3(0.0, 0.0, 1.0));
}

// Multiply 3x3 matrix m and float3 vector v
float3 vdot(float3x3 m, float3 v) {
  return make_float3(m.x.x*v.x + m.x.y*v.y + m.x.z*v.z, m.y.x*v.x + m.y.y*v.y + m.y.z*v.z, m.z.x*v.x + m.z.y*v.y + m.z.z*v.z);
}

// Safe division of float a by float b
float sdivf(float a, float b) {
  if (b == 0.0) return 0.0;
  else return a/b;
}

// Safe division of float3 a by float b
float3 sdivf3(float3 a, float b) {
  return make_float3(sdivf(a.x, b), sdivf(a.y, b), sdivf(a.z, b));
}

// Safe element-wise division of float3 a by float3 b
float3 sdivf33(float3 a, float3 b) {
  return make_float3(sdivf(a.x, b.x), sdivf(a.y, b.y), sdivf(a.z, b.z));
}

// Safe power function raising float a to power float b
float spowf(float a, float b) {
  if (a <= 0.0) return a;
  else return pow(a, b);
}

// Safe power function raising float3 a to power float b
float3 spowf3(float3 a, float b) {
  return make_float3(spowf(a.x, b), spowf(a.y, b), spowf(a.z, b));
}

// Return the hypot or length of float3 a
float hypotf3(float3 a) { return  sqrt(spowf(a.x, 2.0) + spowf(a.y, 2.0) + spowf(a.z, 2.0)); }

// Return the min of float3 a
float fmaxf3(float3 a) { return fmax(a.x, fmax(a.y, a.z)); }

// Return the max of float3 a
float fminf3(float3 a) { return fmin(a.x, fmin(a.y, a.z)); }

// Clamp float3 a to max value mx
float3 clampmaxf3(float3 a, float mx) { return make_float3(fmin(a.x, mx), fmin(a.y, mx), fmin(a.z, mx)); }

// Clamp float3 a to min value mn
float3 clampminf3(float3 a, float mn) { return make_float3(fmax(a.x, mn), fmax(a.y, mn), fmax(a.z, mn)); }

// Clamp each component of float3 a to be between float mn and float mx
float3 clampf3(float3 a, float mn, float mx) { 
  return make_float3(fmin(fmax(a.x, mn), mx), fmin(fmax(a.y, mn), mx), fmin(fmax(a.z, mn), mx));
}


/* OETF Linearization Transfer Functions ---------------------------------------- */

/* float oetf_davinci_intermediate(float x) { */
/*     return ite(x <= 0.02740668, x/10.44426855, exp2(x/0.07329248 - 7.0) - 0.0075); */
/* } */

/* float oetf_filmlight_tlog(float x) { */
/*     return ite(x < 0.075, (x-0.075)/16.184376489665897, exp((x - 0.5520126568606655)/0.09232902596577353) - 0.0057048244042473785); */
/* } */
/* float oetf_arri_logc3(float x) { */
/*     return ite(x < 5.367655*0.010591 + 0.092809, (x - 0.092809)/5.367655, (exp10((x - 0.385537)/0.247190) - 0.052272)/5.555556); */
/* } */

/* float oetf_arri_logc4(float x) { */
/*     return ite(x < -0.7774983977293537, x*0.3033266726886969 - 0.7774983977293537, (exp2(14.0*(x - 0.09286412512218964)/0.9071358748778103 + 6.0) - 64.0)/2231.8263090676883); */
/* } */

/* float oetf_panasonic_vlog(float x) { */
/*     return ite(x < 0.181, (x - 0.125)/5.6, exp10((x - 0.598206)/0.241514) - 0.00873); */
/* } */

/* float oetf_sony_slog3(float x) { */
/*     return ite(x < 171.2102946929/1023.0, (x*1023.0 - 95.0)*0.01125/(171.2102946929 - 95.0), (exp10(((x*1023.0 - 420.0)/261.5))*(0.18 + 0.01) - 0.01)); */
/* } */

/* float oetf_fujifilm_flog(float x) { */
/*     return ite(x < 0.1005377752, (x - 0.092864)/8.735631, (exp10(((x - 0.790453)/0.344676))/0.555556 - 0.009468/0.555556)); */
/* } */


/* float3 linearize(float3 rgb, int tf) { */
/*   if (tf == 0) { // Linear */
/*     return rgb; */
/*   } else if (tf == 1) { // Davinci Intermediate */
/*     rgb.x = oetf_davinci_intermediate(rgb.x); */
/*     rgb.y = oetf_davinci_intermediate(rgb.y); */
/*     rgb.z = oetf_davinci_intermediate(rgb.z); */
/*   } else if (tf == 2) { // Filmlight T-Log */
/*     rgb.x = oetf_filmlight_tlog(rgb.x); */
/*     rgb.y = oetf_filmlight_tlog(rgb.y); */
/*     rgb.z = oetf_filmlight_tlog(rgb.z); */
/*   } else if (tf == 3) { // Arri LogC3 */
/*     rgb.x = oetf_arri_logc3(rgb.x); */
/*     rgb.y = oetf_arri_logc3(rgb.y); */
/*     rgb.z = oetf_arri_logc3(rgb.z); */
/*   } else if (tf == 4) { // Arri LogC4 */
/*     rgb.x = oetf_arri_logc4(rgb.x); */
/*     rgb.y = oetf_arri_logc4(rgb.y); */
/*     rgb.z = oetf_arri_logc4(rgb.z); */
/*   } else if (tf == 5) { // Panasonic V-Log */
/*     rgb.x = oetf_panasonic_vlog(rgb.x); */
/*     rgb.y = oetf_panasonic_vlog(rgb.y); */
/*     rgb.z = oetf_panasonic_vlog(rgb.z); */
/*   } else if (tf == 6) { // Sony S-Log3 */
/*     rgb.x = oetf_sony_slog3(rgb.x); */
/*     rgb.y = oetf_sony_slog3(rgb.y); */
/*     rgb.z = oetf_sony_slog3(rgb.z); */
/*   } else if (tf == 7) { // Fuji F-Log */
/*     rgb.x = oetf_fujifilm_flog(rgb.x); */
/*     rgb.y = oetf_fujifilm_flog(rgb.y); */
/*     rgb.z = oetf_fujifilm_flog(rgb.z); */
/*   } */
/*   return rgb; */
/* } */



/* /\* EOTF Transfer Functions ---------------------------------------- *\/ */

/* float3 eotf_hlg(float3 in_rgb, int inverse) { */
/*   // Aply the HLG Forward or Inverse EOTF. Implements the full ambient surround illumination model */
/*   // ITU-R Rec BT.2100-2 https://www.itu.int/rec/R-REC-BT.2100 */
/*   // ITU-R Rep BT.2390-8: https://www.itu.int/pub/R-REP-BT.2390 */
/*   // Perceptual Quantiser (PQ) to Hybrid Log-Gamma (HLG) Transcoding: https://www.bbc.co.uk/rd/sites/50335f370b5c262af000004/assets/592eea8006d63e5e520090d/BBC_HDRTV_PQ_HLG_Transcode_v2.pdf */

/*     float3 rgb = in_rgb; */
    
/*   const float HLG_Lw = 1000.0; */
/*   // const float HLG_Lb = 0.0; */
/*   const float HLG_Ls = 5.0; */
/*   const float h_a = 0.17883277; */
/*   const float h_b = 1.0 - 4.0*0.17883277; */
/*   const float h_c = 0.5 - h_a*log(4.0*h_a); */
/*   const float h_g = 1.2*spowf(1.111, log2(HLG_Lw/1000.0))*spowf(0.98, log2(fmax(1e-6, HLG_Ls)/5.0)); */
/*   if (inverse == 1) { */
/*     float Yd = 0.2627*rgb.x + 0.6780*rgb.y + 0.0593*rgb.z; */
/*     // HLG Inverse OOTF */
/*     rgb = rgb*spowf(Yd, (1.0 - h_g)/h_g); */
/*     // HLG OETF */
/*     rgb.x = ite(rgb.x <= 1.0/12.0, sqrt(3.0*rgb.x), h_a*log(12.0*rgb.x - h_b) + h_c); */
/*     rgb.y = ite(rgb.y <= 1.0/12.0, sqrt(3.0*rgb.y), h_a*log(12.0*rgb.y - h_b) + h_c); */
/*     rgb.z = ite(rgb.z <= 1.0/12.0, sqrt(3.0*rgb.z), h_a*log(12.0*rgb.z - h_b) + h_c); */
/*   } else { */
/*     // HLG Inverse OETF */
/*     rgb.x = ite(rgb.x <= 0.5, rgb.x*rgb.x/3.0, (exp((rgb.x - h_c)/h_a) + h_b)/12.0); */
/*     rgb.y = ite(rgb.y <= 0.5, rgb.y*rgb.y/3.0, (exp((rgb.y - h_c)/h_a) + h_b)/12.0); */
/*     rgb.z = ite(rgb.z <= 0.5, rgb.z*rgb.z/3.0, (exp((rgb.z - h_c)/h_a) + h_b)/12.0); */
/*     // HLG OOTF */
/*     float Ys = 0.2627*rgb.x + 0.6780*rgb.y + 0.0593*rgb.z; */
/*     rgb = rgb*spowf(Ys, h_g - 1.0); */
/*   } */
/*   return rgb; */
/* } */


/* float3 eotf_pq(float3 rgb, int inverse) { */
/*   /\* Apply the ST-2084 PQ Forward or Inverse EOTF */
/*       ITU-R Rec BT.2100-2 https://www.itu.int/rec/R-REC-BT.2100 */
/*       ITU-R Rep BT.2390-9 https://www.itu.int/pub/R-REP-BT.2390 */
/*       Note: in the spec there is a normalization for peak display luminance.  */
/*       For this function we assume the input is already normalized such that 1.0 = 10,000 nits */
/*   *\/ */
  
/*   // const float Lp = 1.0; */
/*   const float m1 = 2610.0/16384.0; */
/*   const float m2 = 2523.0/32.0; */
/*   const float c1 = 107.0/128.0; */
/*   const float c2 = 2413.0/128.0; */
/*   const float c3 = 2392.0/128.0; */

/*   if (inverse == 1) { */
/*     // rgb /= Lp; */
/*     rgb = spowf3(rgb, m1); */
/*     rgb = spowf3((c1 + c2*rgb)/(1.0 + c3*rgb), m2); */
/*   } else { */
/*     rgb = spowf3(rgb, 1.0/m2); */
/*     rgb = spowf3((rgb - c1)/(c2 - c3*rgb), 1.0/m1); */
/*     // rgb *= Lp; */
/*   } */
/*   return rgb; */
/* } */


float compress_hyperbolic_power(float x, float s, float p)
{
  // Simple hyperbolic compression function https://www.desmos.com/calculator/ofwtcmzc3w
  return spowf(x/(x + s), p);
}

float compress_toe_quadratic(float x, float toe, int inv)
{
  // Quadratic toe compress function https://www.desmos.com/calculator/skk8ahmnws
  if (toe == 0.0) return x;
  if (inv == 0) {
    return spowf(x, 2.0)/(x + toe);
  } else {
    return (x + sqrt(x*(4.0*toe + x)))/2.0;
  }
}

float compress_toe_cubic(float x, float m, float w, int inv)
{
  // https://www.desmos.com/calculator/ubgteikoke
  if (m==1.0) return x;
  float x2 = x*x;
  if (inv == 0) {
    return x*(x2 + m*w)/(x2 + w);
  } else {
    float p0 = x2 - 3.0*m*w;
    float p1 = 2.0*x2 + 27.0*w - 9.0*m*w;
    float p2 = pow(sqrt(x2*p1*p1 - 4*p0*p0*p0)/2.0 + x*p1/2.0, 1.0/3.0);
    return p0/(3.0*p2) + p2/3.0 + x/3.0;
  }
}

float complement_power(float x, float p)
{
  return 1.0 - spowf(1.0 - x, 1.0/p);
}

float sigmoid_cubic(float x, float s)
{
  // Simple cubic sigmoid: https://www.desmos.com/calculator/hzgib42en6
  if (x < 0.0 || x > 1.0) return 1.0;
  return 1.0 + s*(1.0 - 3.0*x*x + 2.0*x*x*x);
}

float contrast_high(float x, float p, float pv, float pv_lx, int inv)
{
  // High exposure adjustment with linear extension
  // https://www.desmos.com/calculator/etjgwyrgad
  const float x0 = 0.18*pow(2.0, pv);
  if (x < x0 || p == 1.0) return x;

  const float o = x0 - x0/p;
  const float s0 = pow(x0, 1.0 - p)/p;
  const float x1 = x0*pow(2.0, pv_lx);
  const float k1 = p*s0*pow(x1, p)/x1;
  const float y1 = s0*pow(x1, p) + o;
  if (inv==1) {
      return ite(x > y1, (x - y1)/k1 + x1, pow((x - o)/s0, 1.0/p));
  } else {
      return ite(x > x1, k1*(x - x1) + y1, s0*pow(x, p) + o);
  }
}

float softplus(float x, float s, float x0, float y0)
{
  // Softplus https://www.desmos.com/calculator/doipi4u0ce
  if (x > 10.0*s + y0 || s < 1e-3) return x;
  float m = 1.0;
  if (fabs(y0) > 1e-6) m = exp(y0/s);
  m = m - exp(x0/s);
  return s*log(fmax(0.0, m + exp(x/s)));
}

float gauss_window(float x, float w)
{
  // Simple gaussian window https://www.desmos.com/calculator/vhr9hstlyk
  float y = x / w;
  return exp(-y*y);
}


float hue_offset(float h, float o)
{
  // Offset hue maintaining 0-2*pi range with modulo
  return fmod(h - o + PI, 2.0*PI) - PI;
}



float3 transform(float p_R, float p_G, float p_B,
                 int in_gamut,
                 int tn_hcon_enable,
                 int tn_lcon_enable,
                 int ptl_enable,
                 int ptm_enable,
                 int brl_enable,
                 int hs_rgb_enable,
                 int hs_cmy_enable,
                 int hc_enable,
                 int cwp,
                 int display_gamut,
                 int eotf,
                 float tn_Lg,
                 float tn_con,
                 float tn_sh,
                 float tn_toe,
                 float tn_off,
                 float tn_hcon,
                 float tn_hcon_pv,
                 float tn_hcon_st,
                 float tn_lcon,
                 float tn_lcon_w,
                 float tn_lcon_pc,
                 float cwp_rng,
                 float rs_sa,
                 float rs_rw,
                 float rs_bw,
                 float pt_r,
                 float pt_g,
                 float pt_b,
                 float pt_rng_low,
                 float pt_rng_high,
                 float ptm_low,
                 float ptm_low_st,
                 float ptm_high,
                 float ptm_high_st,
                 float brl_r,
                 float brl_g,
                 float brl_b,
                 float brl_c,
                 float brl_m,
                 float brl_y,
                 float brl_rng,
                 float hs_r,
                 float hs_g,
                 float hs_b,
                 float hs_rgb_rng,
                 float hs_c,
                 float hs_m,
                 float hs_y,
                 float hc_r,
                 float tn_Lp,
                 float tn_gb,
                 float pt_hdr)

{
  float3 rgb = make_float3(p_R, p_G, p_B);

  float3x3 in_to_xyz;
  if (in_gamut==0) in_to_xyz = identity();
  else if (in_gamut==1) in_to_xyz = matrix_ap0_to_xyz;
  else if (in_gamut==2) in_to_xyz = matrix_ap1_to_xyz;
  else if (in_gamut==3) in_to_xyz = matrix_p3d65_to_xyz;
  else if (in_gamut==4) in_to_xyz = matrix_rec2020_to_xyz;
  else if (in_gamut==5) in_to_xyz = matrix_rec709_to_xyz;
  else if (in_gamut==6) in_to_xyz = matrix_arriwg3_to_xyz;
  else if (in_gamut==7) in_to_xyz = matrix_arriwg4_to_xyz;
  else if (in_gamut==8) in_to_xyz = matrix_redwg_to_xyz;
  else if (in_gamut==9) in_to_xyz = matrix_sonysgamut3_to_xyz;
  else if (in_gamut==10) in_to_xyz = matrix_sonysgamut3cine_to_xyz;
  else if (in_gamut==11) in_to_xyz = matrix_vgamut_to_xyz;
  else if (in_gamut==12) in_to_xyz = matrix_bmdwg_to_xyz;
  else if (in_gamut==13) in_to_xyz = matrix_egamut_to_xyz;
  else if (in_gamut==14) in_to_xyz = matrix_egamut2_to_xyz;
  else if (in_gamut==15) in_to_xyz = matrix_davinciwg_to_xyz;


  // Linearize if a non-linear input oetf / transfer function is selected
  //rgb = linearize(rgb, in_oetf);


  /***************************************************
    Tonescale Constraint Calculations
    https://www.desmos.com/calculator/1c4hzy3bw

    These could be pre-calculated but there is no way to do this in DCTL.
    Anything that is const should be precalculated and not run per-pixel
    --------------------------------------------------*/
  const float ts_x1 = pow(2.0, 6.0*tn_sh + 4.0);
  const float ts_y1 = tn_Lp/100.0;
  const float ts_x0 = 0.18 + tn_off;
  const float ts_y0 = tn_Lg/100.0*(1.0 + tn_gb*log2(ts_y1));
  const float ts_s0 = compress_toe_quadratic(ts_y0, tn_toe, 1);
  const float ts_s10 = ts_x0*(pow(ts_s0, -1.0/tn_con) - 1.0);
  const float ts_m1 = ts_y1/pow(ts_x1/(ts_x1 + ts_s10), tn_con);
  const float ts_m2 = compress_toe_quadratic(ts_m1, tn_toe, 1);
  const float ts_s = ts_x0*(pow(ts_s0/ts_m2, -1.0/tn_con) - 1.0);
  const float ts_dsc = ite(eotf==4, 0.01, ite(eotf==5, 0.1, 100.0/tn_Lp));

  // Lerp from pt_cmp at 100 nits to pt_cmp_hdr at 1000 nits
  const float pt_cmp_Lf = pt_hdr*fmin(1.0, (tn_Lp - 100.0)/900.0);
  // Approximate scene-linear scale at Lp=100 nits
  const float s_Lp100 = ts_x0*(pow((tn_Lg/100.0), -1.0/tn_con) - 1.0);
  const float ts_s1 = ts_s*pt_cmp_Lf + s_Lp100*(1.0 - pt_cmp_Lf);


  // Convert from input gamut into P3-D65
  rgb = vdot(in_to_xyz, rgb);
  rgb = vdot(matrix_xyz_to_p3d65, rgb);


  // Rendering Space: "Desaturate" to control scale of the color volume in the rgb ratios.
  // Controlled by rs_sa (saturation) and red and blue weights (rs_rw and rs_bw)
  float3 rs_w = make_float3(rs_rw, 1.0 - rs_rw - rs_bw, rs_bw);
  float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
  rgb = add_ff3(sat_L*rs_sa, mul_f3f(rgb, (1.0 - rs_sa)));


  // Offset
  rgb = add_f3f(rgb, tn_off);


  /***************************************************
    Contrast Low Module
  --------------------------------------------------*/
  if (tn_lcon_enable) {
    float mcon_m = pow(2.0, -tn_lcon);
    float mcon_w = tn_lcon_w/4.0;
    mcon_w = mcon_w * mcon_w;

    // Normalize for ts_x0 intersection constraint: https://www.desmos.com/calculator/blyvi8t2b2
    const float mcon_cnst_sc = compress_toe_cubic(ts_x0, mcon_m, mcon_w, 1)/ts_x0;
    rgb = mul_f3f(rgb, mcon_cnst_sc);

    // Scale for ratio-preserving midtone contrast
    float mcon_nm = hypotf3(clampminf3(rgb, 0.0))/SQRT3;
    float mcon_sc = (mcon_nm*mcon_nm + mcon_m*mcon_w)/(mcon_nm*mcon_nm + mcon_w);

    if (tn_lcon_pc > 0.0) {
      // Mix between ratio-preserving and per-channel by blending based on distance from achromatic

      // Apply per-channel midtone contrast
      float3 mcon_rgb = rgb;
      mcon_rgb.x = compress_toe_cubic(rgb.x, mcon_m, mcon_w, 0);
      mcon_rgb.y = compress_toe_cubic(rgb.y, mcon_m, mcon_w, 0);
      mcon_rgb.z = compress_toe_cubic(rgb.z, mcon_m, mcon_w, 0);

      // Always use some amount of ratio-preserving method towards gamut boundary
      float mcon_mx = fmaxf3(rgb);
      float mcon_mn = fminf3(rgb);
      float mcon_ch = clampf(1.0 - sdivf(mcon_mn, mcon_mx), 0.0, 1.0);
      mcon_ch = pow(mcon_ch, 4.0*tn_lcon_pc);
      rgb = add_f3f3(mul_f3f(rgb, mcon_sc*mcon_ch), mul_f3f(mcon_rgb, (1.0 - mcon_ch)));
    }
    else { // Just use ratio-preserving
        rgb = mul_f3f(rgb, mcon_sc);
    }
  }


  // Tonescale Norm
  float tsn = hypotf3(clampminf3(rgb, 0.0))/SQRT3;
  // Purity Compression Norm
  float ts_pt = sqrt(fmax(0.0, rgb.x*rgb.x*pt_r + rgb.y*rgb.y*pt_g + rgb.z*rgb.z*pt_b));

  // RGB Ratios
  rgb = sdivf3(clampminf3(rgb, -2.0), tsn);



  // Apply High Contrast
  if (tn_hcon_enable) {
    float hcon_p = pow(2.0, tn_hcon);
    tsn = contrast_high(tsn, hcon_p, tn_hcon_pv, tn_hcon_st, 0);
    ts_pt = contrast_high(ts_pt, hcon_p, tn_hcon_pv, tn_hcon_st, 0);
  }

  // Apply tonescale
  tsn = compress_hyperbolic_power(tsn, ts_s, tn_con);
  ts_pt = compress_hyperbolic_power(ts_pt, ts_s1, tn_con);


  // Simple Cyan-Yellow / Green-Magenta opponent space for calculating smooth achromatic distance and hue angles
  float opp_cy = rgb.x - rgb.z;
  float opp_gm = rgb.y - (rgb.x + rgb.z)/2.0;
  float ach_d = sqrt(fmax(0.0, opp_cy*opp_cy + opp_gm*opp_gm))/SQRT3;

  // Smooth ach_d, normalized so 1.0 doesn't change https://www.desmos.com/calculator/ozjg09hzef
  ach_d = (1.25)*compress_toe_quadratic(ach_d, 0.25, 0);

  // Hue angle, rotated so that red = 0.0
  float hue = fmod(atan2(opp_cy, opp_gm) + PI + 1.10714931, 2.0*PI);

  // RGB Hue Angles
  // Wider than CMY by default. R towards M, G towards Y, B towards C
  float3 ha_rgb = make_float3(
    gauss_window(hue_offset(hue, 0.1), 0.9),
    gauss_window(hue_offset(hue, 4.3), 0.9),
    gauss_window(hue_offset(hue, 2.3), 0.9));

  // CMY Hue Angles
  // Exact alignment to Cyan/Magenta/Yellow secondaries would be PI, PI/3 and -PI/3, but
  // we customize these a bit for creative purposes: M towards B, Y towards G, C towards G
  float3 ha_cmy = make_float3(
    gauss_window(hue_offset(hue, 3.3), 0.6),
    gauss_window(hue_offset(hue, 1.3), 0.6),
    gauss_window(hue_offset(hue, -1.2), 0.6));


  // Purity Compression Range: https://www.desmos.com/calculator/8ynarg1uxk
  float ts_pt_cmp = 1.0 - pow(ts_pt, 1.0/pt_rng_low);

  float pt_rng_high_f = fmin(1.0, ach_d/1.2);
  pt_rng_high_f = pt_rng_high_f * pt_rng_high_f;
  pt_rng_high_f = ite(pt_rng_high < 1.0, 1.0 - pt_rng_high_f, pt_rng_high_f);
  ts_pt_cmp = pow(ts_pt_cmp, pt_rng_high)*(1.0 - pt_rng_high_f) + ts_pt_cmp*pt_rng_high_f;


  /***************************************************
    Brilliance
  --------------------------------------------------*/
  float brl_f = 1.0;
  if (brl_enable) {
    brl_f = -brl_r*ha_rgb.x - brl_g*ha_rgb.y - brl_b*ha_rgb.z - brl_c*ha_cmy.x - brl_m*ha_cmy.y - brl_y*ha_cmy.z;
    brl_f = (1.0 - ach_d)*brl_f + 1.0 - brl_f;
    brl_f = softplus(brl_f, 0.25, -100.0, 0.0); // Protect against over-darkening
    
    // Limit Brilliance adjustment by tonescale
    float brl_ts = ite(brl_f > 1.0, 1.0 - ts_pt, ts_pt); // Limit by inverse tonescale if positive Brilliance adjustment
    float brl_lim = spowf(brl_ts, 1.0 - brl_rng);
    brl_f = brl_f*brl_lim + 1.0 - brl_lim;
    brl_f = fmax(0.0, fmin(2.0, brl_f)); // protect for shadow grain
  }



  /***************************************************
    Mid-Range Purity
      This boosts mid-range purity on the low end
      and reduces mid-range purity on the high end
  --------------------------------------------------*/
  float ptm_sc = 1.0;
  if (ptm_enable) {
    // Mid Purity Low
    float ptm_ach_d = complement_power(ach_d, ptm_low_st);
    ptm_sc = sigmoid_cubic(ptm_ach_d, ptm_low*(1.0 - ts_pt));

    // Mid Purity High
    ptm_ach_d = complement_power(ach_d, ptm_high_st)*(1.0 - ts_pt) + ach_d*ach_d*ts_pt;
    ptm_sc = ptm_sc * sigmoid_cubic(ptm_ach_d, ptm_high*ts_pt);
    ptm_sc = fmax(0.0, ptm_sc); // Ensure no negative scale
  }


  // Premult hue angles for Hue Contrast and Hue Shift
  ha_rgb = mul_f3f(ha_rgb, ach_d);
  ha_cmy = mul_f3f(ha_cmy, (1.5)*compress_toe_quadratic(ach_d, 0.5, 0)); // Stronger smoothing for CMY hue shift


  /***************************************************
    Hue Contrast R
  --------------------------------------------------*/
  if (hc_enable) {
    float hc_ts = 1.0 - ts_pt;
    // Limit high purity on bottom end and low purity on top end by ach_d.
    // This helps reduce artifacts and over-saturation.
    float hc_c = (1.0 - ach_d)*hc_ts + ach_d*(1.0 - hc_ts);
    hc_c = hc_c * ha_rgb.x;
    hc_ts = hc_ts * hc_ts;
    // Bias contrast based on tonescale using Lift/Mult: https://www.desmos.com/calculator/gzbgov62hl
    float hc_f = hc_r*(hc_c - 2.0*hc_c*hc_ts) + 1.0;
    rgb = make_float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);
  }



  /***************************************************
  Hue Shift
  --------------------------------------------------*/

  // Hue Shift RGB by purity compress tonescale, shifting more as intensity increases
  if (hs_rgb_enable) {
    float3 hs_rgb = mul_f3f(ha_rgb, pow(ts_pt, 1.0/hs_rgb_rng));
    float3 hsf = make_float3(hs_rgb.x*hs_r, hs_rgb.y*-hs_g, hs_rgb.z*-hs_b);
    hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    rgb = add_f3f3(rgb, hsf);
  }

  // Hue Shift CMY by tonescale, shifting less as intensity increases
  if (hs_cmy_enable) {
    float3 hs_cmy = mul_f3f(ha_cmy, (1.0 - ts_pt));
    float3 hsf = make_float3(hs_cmy.x*-hs_c, hs_cmy.y*hs_m, hs_cmy.z*hs_y);
    hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
    rgb = add_f3f3(rgb, hsf);
  }

  // Apply brilliance
  rgb = mul_f3f(rgb, brl_f);

  // Apply purity compression and mid purity
  ts_pt_cmp = ts_pt_cmp * ptm_sc;
  rgb = add_f3f(mul_f3f(rgb, ts_pt_cmp), 1.0 - ts_pt_cmp);

  // Inverse Rendering Space
  sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
  rgb = mul_f3f(sub_ff3(sat_L*rs_sa, rgb), 1.0/(rs_sa - 1.0));

  // Convert to final display gamut
  float3 cwp_rgb = rgb;
  if (display_gamut==0) {
    if (cwp==1) cwp_rgb = vdot(matrix_p3_to_rec709_d60, rgb);
    if (cwp==2) cwp_rgb = vdot(matrix_p3_to_rec709_d55, rgb);
    if (cwp==3) cwp_rgb = vdot(matrix_p3_to_rec709_d50, rgb);
    rgb = vdot(matrix_p3_to_rec709_d65, rgb);
    if (cwp==0) cwp_rgb = rgb;
  }
  else if (display_gamut>=1) {
    if (cwp==1) cwp_rgb = vdot(matrix_p3_to_p3_d60, rgb);
    if (cwp==2) cwp_rgb = vdot(matrix_p3_to_p3_d55, rgb);
    if (cwp==3) cwp_rgb = vdot(matrix_p3_to_p3_d50, rgb);
  }

  // Mix between Creative Whitepoint and D65 by tsn
  float cwp_f = pow(tsn, 1.0 - cwp_rng);
  rgb = add_f3f3(mul_f3f(cwp_rgb, cwp_f), mul_f3f(rgb, (1.0 - cwp_f)));

  // Purity Compress Low
  if (ptl_enable) {
    float sum0 = softplus(rgb.x, 0.2, -100.0, -0.3) + rgb.y + softplus(rgb.z, 0.2, -100.0, -0.3);
    rgb.x = softplus(rgb.x, 0.04, -0.3, 0.0);
    rgb.y = softplus(rgb.y, 0.06, -0.3, 0.0);
    rgb.z = softplus(rgb.z, 0.01, -0.05, 0.0);

    float ptl_norm = fmin(1.0, sdivf(sum0, rgb.x + rgb.y + rgb.z));
    rgb = mul_f3f(rgb, ptl_norm);
  }

  // Final tonescale adjustments
  tsn = tsn * ts_m2; // scale for inverse toe
  tsn = compress_toe_quadratic(tsn, tn_toe, 0);
  tsn = tsn * ts_dsc; // scale for display encoding

  // Return from RGB ratios
  rgb = mul_f3f(rgb, tsn);

  // Clamp
  //if (_clamp)
  rgb = clampf3(rgb, 0.0, 1.0);

  // Rec.2020 (P3 Limited)
  if (display_gamut==2) {
    rgb = clampminf3(rgb, 0.0);
    rgb = vdot(matrix_p3_to_rec2020, rgb);
  }


  /* // Apply inverse Display EOTF */
  /* float eotf_p = 2.0 + eotf * 0.2; */
  /* if ((eotf > 0) && (eotf < 4)) { */
  /*   rgb = spowf3(rgb, 1.0/eotf_p); */
  /* } else if (eotf == 4) { */
  /*   rgb = eotf_pq(rgb, 1); */
  /* } else if (eotf == 5) { */
  /*   rgb = eotf_hlg(rgb, 1); */
  /* } */


  return rgb;
}

//-----------------------------------------------------------------------------

// @ART-label: "OpenDRT"
// @ART-colorspace: "rec2020"
// @ART-lut: 48

// @ART-param: ["tn_Lp", "Display Peak Luminance", 100, 1000, 100, 0.1]
// @ART-param: ["tn_gb", "HDR Grey Boost", 0, 1, 0.13, 0.01]
// @ART-param: ["pt_hdr", "HDR Purity", 0, 1, 0.5, 0.01]
// @ART-param: ["look_preset", "Look Preset", ["Default", "Colorful", "Umbra", "Base"], 0]
// @ART-param: ["tonescale_preset", "Tonescale Preset", ["Use Look Preset", "High-Contrast", "Low-Contrast", "ACES-1.x", "ACES-2.0", "Marvelous Tonescape", "Arriba Tonecall", "DaGrinchi Tonegroan", "Aery Tonescale", "Umbra Tonescale"], 0]
// @ART-param: ["_cwp", "Creative White", ["D65", "D60", "D55", "D50", "Use Look Preset"], 0]
// @ART-param: ["_cwp_rng", "Creative White Range", 0, 1, 0.5, 0.01]
// @ART-param: ["display_gamut", "Display gamut", ["Rec.709", "P3 D65", "Rec.2020"]]

void ART_main(varying float r, varying float g, varying float b,
              output varying float rout,
              output varying float gout,
              output varying float bout,
              float tn_Lp, float tn_gb, float pt_hdr,
              int look_preset, int tonescale_preset,
              int _cwp, float _cwp_rng,
              int display_gamut)
{
  // **************************************************
  // Parameter Setup
  // --------------------------------------------------

    int tn_hcon_enable;
    int tn_lcon_enable;
    int ptl_enable;
    int ptm_enable;
    int brl_enable;
    int hs_rgb_enable;
    int hs_cmy_enable;
    int hc_enable;
    int cwp;
    int display_gamut;
    int eotf;
    float tn_Lg;
    float tn_con;
    float tn_sh;
    float tn_toe;
    float tn_off;
    float tn_hcon;
    float tn_hcon_pv;
    float tn_hcon_st;
    float tn_lcon;
    float tn_lcon_w;
    float tn_lcon_pc;
    float cwp_rng;
    float rs_sa;
    float rs_rw;
    float rs_bw;
    float pt_r;
    float pt_g;
    float pt_b;
    float pt_rng_low;
    float pt_rng_high;
    float ptm_low;
    float ptm_low_st;
    float ptm_high;
    float ptm_high_st;
    float brl_r;
    float brl_g;
    float brl_b;
    float brl_c;
    float brl_m;
    float brl_y;
    float brl_rng;
    float hs_r;
    float hs_g;
    float hs_b;
    float hs_rgb_rng;
    float hs_c;
    float hs_m;
    float hs_y;
    float hc_r;

  // Look presets to go after
  if (look_preset==0) { // Default
      tn_Lg = 11.1;
      tn_con = 1.4;
      tn_sh = 0.5;
      tn_toe = 0.003;
      tn_off = 0.005;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 1;
      tn_lcon = 1.0;
      tn_lcon_w = 0.5;
      tn_lcon_pc = 1.0;
      cwp = 0;
      cwp_rng = 0.5;
      rs_sa = 0.35;
      rs_rw = 0.25;
      rs_bw = 0.55;
      pt_r = 0.5;
      pt_g = 2.0;
      pt_b = 2.0;
      pt_rng_low = 0.2;
      pt_rng_high = 0.8;
      ptl_enable = 1;
      ptm_enable = 1;
      ptm_low = 0.2;
      ptm_low_st = 0.5;
      ptm_high = -0.8;
      ptm_high_st = 0.3;
      brl_enable = 1;
      brl_r = -0.5;
      brl_g = -0.4;
      brl_b = -0.2;
      brl_c = 0.0;
      brl_m = 0.0;
      brl_y = 0.0;
      brl_rng = 0.66;
      hs_rgb_enable = 1;
      hs_r = 0.35;
      hs_g = 0.25;
      hs_b = 0.5;
      hs_rgb_rng = 0.6;
      hs_cmy_enable = 1;
      hs_c = 0.2;
      hs_m = 0.2;
      hs_y = 0.2;
      hc_enable = 1;
      hc_r = 0.6;
  }
  else if (look_preset==1) { // Colorful
      tn_Lg = 11.1;
      tn_con = 1.3;
      tn_sh = 0.5;
      tn_toe = 0.005;
      tn_off = 0.005;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 1;
      tn_lcon = 0.75;
      tn_lcon_w = 1.0;
      tn_lcon_pc = 1.0;
      cwp = 0;
      cwp_rng = 0.5;
      rs_sa = 0.35;
      rs_rw = 0.15;
      rs_bw = 0.55;
      pt_r = 0.5;
      pt_g = 0.8;
      pt_b = 0.5;
      pt_rng_low = 0.25;
      pt_rng_high = 0.5;
      ptl_enable = 1;
      ptm_enable = 1;
      ptm_low = 0.5;
      ptm_low_st = 0.5;
      ptm_high = -0.8;
      ptm_high_st = 0.3;
      brl_enable = 1;
      brl_r = -0.55;
      brl_g = -0.5;
      brl_b = 0.0;
      brl_c = 0.0;
      brl_m = 0.0;
      brl_y = 0.1;
      brl_rng = 0.5;
      hs_rgb_enable = 1;
      hs_r = 0.4;
      hs_g = 0.6;
      hs_b = 0.5;
      hs_rgb_rng = 0.6;
      hs_cmy_enable = 1;
      hs_c = 0.2;
      hs_m = 0.1;
      hs_y = 0.2;
      hc_enable = 1;
      hc_r = 0.8;
  }
  else if (look_preset==2) { // Umbra
      tn_Lg = 6.0;
      tn_con = 1.8;
      tn_sh = 0.5;
      tn_toe = 0.001;
      tn_off = 0.015;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 1;
      tn_lcon = 1.0;
      tn_lcon_w = 1.0;
      tn_lcon_pc = 1.0;
      cwp = 3;
      cwp_rng = 0.8;
      rs_sa = 0.45;
      rs_rw = 0.1;
      rs_bw = 0.35;
      pt_r = 0.1;
      pt_g = 0.4;
      pt_b = 2.5;
      pt_rng_low = 0.2;
      pt_rng_high = 0.8;
      ptl_enable = 1;
      ptm_enable = 1;
      ptm_low = 0.4;
      ptm_low_st = 0.5;
      ptm_high = -0.8;
      ptm_high_st = 0.3;
      brl_enable = 1;
      brl_r = -0.7;
      brl_g = -0.6;
      brl_b = -0.2;
      brl_c = 0.0;
      brl_m = -0.25;
      brl_y = 0.1;
      brl_rng = 0.9;
      hs_rgb_enable = 1;
      hs_r = 0.4;
      hs_g = 0.8;
      hs_b = 0.4;
      hs_rgb_rng = 1.0;
      hs_cmy_enable = 1;
      hs_c = 1.0;
      hs_m = 0.6;
      hs_y = 1.0;
      hc_enable = 1;
      hc_r = 0.8;
  }
  else if (look_preset==3) { // Base
      tn_Lg = 11.1;
      tn_con = 1.4;
      tn_sh = 0.5;
      tn_toe = 0.003;
      tn_off = 0.0;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 0;
      tn_lcon = 0.0;
      tn_lcon_w = 0.5;
      tn_lcon_pc = 1.0;
      cwp = 0;
      cwp_rng = 0.5;
      rs_sa = 0.35;
      rs_rw = 0.25;
      rs_bw = 0.5;
      pt_r = 1.0;
      pt_g = 2.0;
      pt_b = 2.5;
      pt_rng_low = 0.25;
      pt_rng_high = 0.25;
      ptl_enable = 1;
      ptm_enable = 0;
      ptm_low = 0.0;
      ptm_low_st = 0.5;
      ptm_high = 0.0;
      ptm_high_st = 0.3;
      brl_enable = 0;
      brl_r = 0.0;
      brl_g = 0.0;
      brl_b = 0.0;
      brl_c = 0.0;
      brl_m = 0.0;
      brl_y = 0.0;
      brl_rng = 0.5;
      hs_rgb_enable = 0;
      hs_r = 0.0;
      hs_g = 0.0;
      hs_b = 0.0;
      hs_rgb_rng = 0.5;
      hs_cmy_enable = 0;
      hs_c = 0.0;
      hs_m = 0.0;
      hs_y = 0.0;
      hc_enable = 0;
      hc_r = 0.0;
  }

  /*----------- TONESCALE PRESETS -----------------*/
  if (tonescale_preset==0) { // Do nothing and use tonescale settings from the look preset
  }
  if (tonescale_preset==1) { // High-Contrast
      tn_Lg = 11.1;
      tn_con = 1.4;
      tn_sh = 0.5;
      tn_toe = 0.003;
      tn_off = 0.005;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 1;
      tn_lcon = 1.0;
      tn_lcon_w = 0.5;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==2) { // Low-Contrast
      tn_Lg = 11.1;
      tn_con = 1.4;
      tn_sh = 0.5;
      tn_toe = 0.003;
      tn_off = 0.005;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 0;
      tn_lcon = 0.0;
      tn_lcon_w = 0.5;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==3) { // ACES-1.x
      tn_Lg = 10.0;
      tn_con = 1.0;
      tn_sh = 0.245;
      tn_toe = 0.02;
      tn_off = 0.0;
      tn_hcon_enable = 1;
      tn_hcon = 0.55;
      tn_hcon_pv = 0.0;
      tn_hcon_st = 2.0;
      tn_lcon_enable = 1;
      tn_lcon = 1.13;
      tn_lcon_w = 1.0;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==4) { // ACES-2.0
      tn_Lg = 10.0;
      tn_con = 1.15;
      tn_sh = 0.5;
      tn_toe = 0.04;
      tn_off = 0.0;
      tn_hcon_enable = 0;
      tn_hcon = 1.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 1.0;
      tn_lcon_enable = 0;
      tn_lcon = 1.0;
      tn_lcon_w = 0.6;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==5) { // Marvelous Tonescape
      tn_Lg = 6.0;
      tn_con = 1.5;
      tn_sh = 0.5;
      tn_toe = 0.003;
      tn_off = 0.01;
      tn_hcon_enable = 1;
      tn_hcon = 0.25;
      tn_hcon_pv = 0.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 1;
      tn_lcon = 1.0;
      tn_lcon_w = 1.0;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==6) { // Arriba Tonecall
      tn_Lg = 11.1;
      tn_con = 1.05;
      tn_sh = 0.5;
      tn_toe = 0.1;
      tn_off = 0.015;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 0.0;
      tn_hcon_st = 2.0;
      tn_lcon_enable = 1;
      tn_lcon = 2.0;
      tn_lcon_w = 0.2;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==7) { // DaGrinchi Tonegroan
      tn_Lg = 10.42;
      tn_con = 1.2;
      tn_sh = 0.5;
      tn_toe = 0.02;
      tn_off = 0.0;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 1.0;
      tn_lcon_enable = 0;
      tn_lcon = 0.0;
      tn_lcon_w = 0.6;
      tn_lcon_pc = 1.0;
  }
  else if (tonescale_preset==8) { // Aery Tonescale
      tn_Lg = 11.1;
      tn_con = 1.15;
      tn_sh = 0.5;
      tn_toe = 0.04;
      tn_off = 0.006;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 0.0;
      tn_hcon_st = 0.5;
      tn_lcon_enable = 1;
      tn_lcon = 0.5;
      tn_lcon_w = 2.0;
      tn_lcon_pc = 0.5;
  }
  else if (tonescale_preset==9) { // Umbra Tonescale
      tn_Lg = 6.0;
      tn_con = 1.8;
      tn_sh = 0.5;
      tn_toe = 0.001;
      tn_off = 0.015;
      tn_hcon_enable = 0;
      tn_hcon = 0.0;
      tn_hcon_pv = 1.0;
      tn_hcon_st = 4.0;
      tn_lcon_enable = 1;
      tn_lcon = 1.0;
      tn_lcon_w = 1.0;
      tn_lcon_pc = 1.0;
  }
  
  /*----------- CREATIVE WHITE PRESETS -----------------*/
  // Use user creative whitepoint settings if custom whitepoint specified, otherwise use the one from the look preset
  if (_cwp!=4) cwp_rng = _cwp_rng; 
  if (_cwp==0) cwp = 0; // D65
  else if (_cwp==1) cwp = 1; // D60
  else if (_cwp==2) cwp = 2; // D55
  else if (_cwp==3) cwp = 3; // D50

  float3 rgb = transform(r, g, b,
                         4,
                         tn_hcon_enable,
                         tn_lcon_enable,
                         ptl_enable,
                         ptm_enable,
                         brl_enable,
                         hs_rgb_enable,
                         hs_cmy_enable,
                         hc_enable,
                         cwp,
                         display_gamut,
                         eotf,
                         tn_Lg,
                         tn_con,
                         tn_sh,
                         tn_toe,
                         tn_off,
                         tn_hcon,
                         tn_hcon_pv,
                         tn_hcon_st,
                         tn_lcon,
                         tn_lcon_w,
                         tn_lcon_pc,
                         cwp_rng,
                         rs_sa,
                         rs_rw,
                         rs_bw,
                         pt_r,
                         pt_g,
                         pt_b,
                         pt_rng_low,
                         pt_rng_high,
                         ptm_low,
                         ptm_low_st,
                         ptm_high,
                         ptm_high_st,
                         brl_r,
                         brl_g,
                         brl_b,
                         brl_c,
                         brl_m,
                         brl_y,
                         brl_rng,
                         hs_r,
                         hs_g,
                         hs_b,
                         hs_rgb_rng,
                         hs_c,
                         hs_m,
                         hs_y,
                         hc_r,
                         tn_Lp,
                         tn_gb,
                         pt_hdr);

  float outscale = tn_Lp / 100.0;

  rgb = mul_f3f(rgb, outscale);
  if (display_gamut == 0) {
      rgb = vdot(matrix_xyz_to_rec2020, vdot(matrix_rec709_to_xyz, rgb));
  } else if (display_gamut == 1) {
      rgb = vdot(matrix_xyz_to_rec2020, vdot(matrix_p3d65_to_xyz, rgb));
  }

  rout = rgb.x;
  gout = rgb.y;
  bout = rgb.z;
}
