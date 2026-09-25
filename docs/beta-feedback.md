# Dima's beta feedback

## Implemented in this pass

- Removing the current photo selects its next neighbor, or its previous neighbor when removing the last photo. Removing another photo preserves selection.
- Delete and Backspace remove the selected library photo. Editing fields, dialogs, modifier combinations, composition, and repeated keydown events are excluded.
- One floating adjustment panel contains Exposure, Rendering, White balance, and Detail. Toolbar buttons navigate to groups; scrolling tracks the active group. Settings is visually separated. Exposure and contrast are adjacent. The visible surface ends after Detail, with transparent scroll travel only far enough to center that last group in the viewport. Wheel input stays with the panel throughout its column, including the area below the shortened surface; the column runs between the top bar and the bottom HUD row so it never covers Export/Process, the zoom controls or the filmstrip, and the minimap steps aside while a panel is open. Toolbar navigation centers short groups (including Exposure); taller groups align to the top. Navigation within an already-open panel scrolls smoothly; opening and reduced-motion navigation remain immediate. Group headers stick while their controls scroll, replacing the generic Adjustments heading; the close button stays available at the top.
- Histogram source/channel controls live in the histogram widget instead of the adjustment toolbar.
- Manual zoom reaches 75% of fit-to-viewport scale. Fit uses the entire viewport. Opening, closing, or redrawing floating controls must not change photo scale or position.
- Vertical mouse-wheel input scrolls the thumbnail strip horizontally; horizontal gestures retain native behavior and scrolling does not select another photo. Confirm this interpretation with Dima.

## Remaining work

- Crop/rotate/straighten: a separate toolbar button and image editing mode, not an adjustment group. Confirm straightening, 90-degree rotation, and aspect-ratio needs. Must work consistently in preview and export.
- Batch export: establish whether Export all is sufficient or a subset is essential. Preserve individual edits with shared output settings. Needs a reliable export queue, including processing readiness and failure handling.
- Contrast curve: obtain a source photo and reproduction/recording; distinguish dragging behavior from image rendering or expectations about curve semantics.
- HDR: obtain browser/OS/display details and OS HDR state. Current Skip dismisses a brightness-calibration request after HDR was enabled; it is not an SDR choice. Investigate detection and revise the interaction deliberately.
- Favicon: retrieve Dima's existing asset, add it, and verify both beta and stable deployment paths.

## Separate design spike

Audit translucent surface contrast across widgets, panels, toolbars, and dialogs. Explore a stronger dark surface base, semantic text/icon roles, and hierarchy conveyed through layout and typography. Validate over bright, dark, gray, and busy backgrounds. No color-system changes in this pass.

## Positive feedback awaiting verification

Dima liked browser Ctrl +/− behavior and Fit / 100% / 200%. Independently verify intended behavior before treating that feedback as a regression specification. The new zoom tests cover the explicitly agreed manual lower bound and stationary content during panel changes; they do not certify every existing zoom interaction.
