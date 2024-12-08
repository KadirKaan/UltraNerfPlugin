#include "NeRFRenderer.h"

#include "NeRFUtils.h"

using namespace torch::indexing;

NeRFRenderer::NeRFRenderer(NeRFModel &model, int H, int W, float focal)
    : model(model), H(H), W(W), focal(focal) {}

torch::Tensor NeRFRenderer::render(const torch::Tensor &pose, bool randomize,
                                   float startDistance, float endDistance,
                                   int nSamples, int batchSize) const
{
    auto rays = getRays(pose.to(model.getDevice()));
    return renderRays(rays, randomize, startDistance, endDistance, nSamples,
                      batchSize);
}

NeRFRenderer::RayData NeRFRenderer::getRays(const torch::Tensor &pose) const
{
    // Generate pixel indices along image width (i) and height (j)
    auto i = torch::arange(W, torch::dtype(torch::kFloat32)).to(model.getDevice());
    auto j = torch::arange(H, torch::dtype(torch::kFloat32)).to(model.getDevice());
    auto grid = torch::meshgrid({i, j}, "xy");
    auto ii = grid[0];
    auto jj = grid[1];

    // Compute the direction vector for each pixel in the image plane
    auto dirs = torch::stack({(ii - W * 0.5) / focal, -(jj - H * 0.5) / focal,
                              -torch::ones_like(ii)},
                             -1);

    // Transform the direction vectors from the camera's local coordinate system
    // to the global coordinate system
    auto raysD = torch::sum(dirs.index({"...", None, Slice()}) *
                                pose.index({Slice(0, 3), Slice(0, 3)}),
                            -1);
    // Get the origin of the rays from the pose
    auto raysO = pose.index({Slice(0, 3), -1}).expand(raysD.sizes());

    return std::make_tuple(raysO, raysD);
}

torch::Tensor NeRFRenderer::renderRays(const RayData &rays, bool randomize,
                                       float startDistance,
                                       float endDistance, int nSamples,
                                       int batchSize) const
{
    // Unpack the ray origins and directions
    auto raysO = std::get<0>(rays);
    auto raysD = std::get<1>(rays);

    // Compute 3D query points
    auto zVals =
        torch::linspace(startDistance, endDistance, nSamples, model.getDevice())
            .reshape({1, 1, nSamples})
            .expand({H, W, nSamples})
            .clone();
    if (randomize)
    {
        zVals += torch::rand({H, W, nSamples}, model.getDevice()) *
                 (startDistance - endDistance) / nSamples;
    }
    auto pts = raysO.unsqueeze(-2) + raysD.unsqueeze(-2) * zVals.unsqueeze(-1);

    // Encode points
    auto ptsFlat = pts.view({-1, 3});
    auto ptsEmbedded = model.addPositionalEncoding(ptsFlat);

    // Batch-process points
    int nPts = ptsFlat.size(0);
    torch::Tensor raw;
    for (int i = 0; i < nPts; i += batchSize)
    {
        auto batch = ptsEmbedded.slice(0, i, std::min(i + batchSize, nPts));
        auto batchRaw = model.forward(batch);
        if (i == 0)
        {
            raw = batchRaw;
        }
        else
        {
            raw = torch::cat({raw, batchRaw}, 0);
        }
    }
    raw = raw.view({H, W, nSamples, 4});

    // Get volume colors and opacities
    auto rgb = torch::sigmoid(raw.index({"...", Slice(0, 3)}));
    auto sigmaA = torch::relu(raw.index({"...", 3}));

    // Render volume
    auto dists = torch::cat({zVals.index({"...", Slice(1, None)}) -
                                 zVals.index({"...", Slice(None, -1)}),
                             torch::full({1}, 1e10, model.getDevice()).expand({H, W, 1})},
                            -1);
    auto alpha = 1.0 - torch::exp(-sigmaA * dists);
    auto weights = torch::cumprod(1.0 - alpha + 1e-10, -1);
    weights = alpha * torch::cat({torch::ones({H, W, 1}, model.getDevice()),
                                  weights.index({"...", Slice(None, -1)})},
                                 -1);

    auto rgbMap = torch::sum(weights.unsqueeze(-1) * rgb, -2);
    auto depthMap = torch::sum(weights * zVals, -1);
    auto accMap = torch::sum(weights, -1);

    return rgbMap;
}