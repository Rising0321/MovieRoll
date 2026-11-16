# Examples

All outputs below pass `scripts/validate_prompt.py`.

## 1. Single mode — subject moving through space, tracking camera

**Input:** `A golden retriever chases a frisbee on the beach, camera tracking it.`

**Output:**

A golden retriever with a red collar runs along a wide sandy beach at golden hour, chasing a bright orange frisbee that skims low over the sand ahead of it. The dog has a thick, wavy golden coat that glows in the low sun, floppy ears, a dark wet nose, and a red nylon collar with a small silver tag. Initially, the golden retriever lopes along the damp sand near the waterline at a natural, easy pace, eyes fixed on the orange frisbee. Then it lunges forward, snaps the frisbee out of the air, and keeps running with the disc in its mouth, paws kicking up small sprays of wet sand with every stride. Throughout, the dog continues its steady, rhythmic run along the shoreline, ears bouncing and tail streaming behind it, occasionally tossing the frisbee up with a flick of its head and catching it again without breaking stride. Beside it, gentle waves roll steadily onto the shore with a soft, rhythmic hiss, leaving thin sheets of foam that slide back into the sea, while a few seagulls glide across the pale peach sky in the distance. The golden retriever's glowing coat and red collar stay clearly visible against the blue-gray water. The camera tracks the golden retriever, continuously keeping the golden retriever centered. Eye-level medium-wide shot. Warm golden-hour backlight traces a soft rim around the dog's fur, with long shadows stretching across the sand and a harmonious warm-orange and cool-blue palette. Sharp focus, fine fur detail. Joyful and carefree. Cinematic photography realistic style.

**What to notice**
- The dog travels along the beach, so the tracking behavior is kept, stated once with the canonical tracking sentence; a medium-wide shot suits the large motion range.
- The chase is written as an ongoing run with a repeating toss-and-catch, not a single catch that ends the scene; the pace stays natural ("easy," "steady").
- No thrower was invented: the frisbee's motion is described without adding a new person.
- Identity is re-anchored ("glowing coat and red collar") before the technical block; the style tag is last.

## 2. Sequential mode — two events arriving one at a time

### Event 1

User message sent to the model:

```text
PREVIOUS EVENTS (already finalized): None — this is the first event in the sequence.

NEW EVENT TO REWRITE:
An old fisherman mends a fishing net on a wooden dock at dawn.

Rewrite ONLY the new event above into one complete prompt, following the Multi-Event Streaming Mode rules.
```

Output (becomes finalized history):

An elderly fisherman sits on a weathered wooden dock at dawn, patiently mending a large fishing net draped across his lap beside a calm harbor. He has a short white beard, deeply tanned and wrinkled skin, and wears a faded blue knit cap, a thick gray wool sweater with rolled sleeves, and olive rubber boots. Initially, the elderly fisherman lifts a torn section of the green nylon net toward the soft light, squinting as he inspects the broken mesh. Then he threads a wooden netting needle through the gap, pulls the twine taut with a steady tug, and ties a small knot with practiced fingers. Throughout, he repeats this calm rhythm of threading, pulling, and knotting, shifting the net across his knees section by section while his shoulders rise and fall with each unhurried breath. Behind him, small fishing boats bob gently at their moorings on the pale, glassy water, and a thin layer of mist drifts low over the harbor as a few gulls glide high above, their faint cries carrying across the water. The white-bearded fisherman in the faded blue knit cap remains the calm focal point of the dock. The camera remains static, keeping the elderly fisherman centered. Eye-level medium-wide shot. Soft pink-and-gold dawn light falls from the side, gently modeling the wrinkles of his face and the fibers of the net, with cool blue shadows on the water. Sharp focus, refined texture. Quiet and contemplative. Cinematic photography realistic style.

### Event 2

User message sent to the model:

```text
PREVIOUS EVENTS (already finalized, read-only context — do NOT rewrite or repeat these):
Event 1 (already rewritten, finalized): An elderly fisherman sits on a weathered wooden dock at dawn, ... Cinematic photography realistic style.

NEW EVENT TO REWRITE:
He looks up as a small boy runs over carrying a bucket of fish.

Rewrite ONLY the new event above into one complete prompt, following the Multi-Event Streaming Mode rules.
```

(In a real request, Event 1 is included in full, not abbreviated.)

Output:

The elderly fisherman with a short white beard and a faded blue knit cap pauses his net-mending on the weathered wooden dock at dawn as a small boy carrying a bucket of fish approaches along the planks. The fisherman still wears his thick gray wool sweater with rolled sleeves and olive rubber boots, the green nylon net resting across his knees. The small boy, about seven years old, has messy black hair and wears a yellow raincoat, blue shorts, and red rubber boots, gripping a dented metal bucket of glistening silver fish with both hands. Initially, the elderly fisherman lifts his head from the net, his weathered face softening into a warm smile. Then the small boy trots in from the left side of the frame at a natural pace, the bucket swaying and water sloshing over its rim, and slows beside the fisherman. As this continues, the boy tips the bucket to show the fish while the fisherman leans forward, nodding and pointing at the catch with a calloused finger, the two exchanging lively gestures. Behind them, small fishing boats bob gently at their moorings on the pale, glassy harbor as thin mist drifts low over the water. Throughout, the white-bearded fisherman in the faded blue knit cap and the boy in the yellow raincoat stay close together at the heart of the dock. The camera remains static, keeping the elderly fisherman and the small boy centered. Eye-level medium-wide shot. Soft pink-and-gold dawn side light, cool blue shadows on the water. Warm and tender. Cinematic photography realistic style.

**What to notice**
- The input's opening "He" became the full established noun phrase ("The elderly fisherman with a short white beard and a faded blue knit cap").
- The fisherman stays in place, so event 1 uses a static camera; event 2 keeps the same setup (eye-level medium-wide, static) because nothing in the new event requires tracking, and the boy enters the established frame instead of the camera reframing.
- The fisherman's clothing, the net, the dock, the dawn light, and the harbor are carried over unchanged.
- The event gets a full-length budget even though it has prior context.

## 3. Batch mode — output layout

The same two events rewritten in one batch call (user message `Event 1: ...` / `Event 2: ...`) must come back as:

```text
<full prompt for event 1, one paragraph>
@@@EVENT_BREAK@@@
<full prompt for event 2, one paragraph>
```

Validate with `python scripts/validate_prompt.py response.txt --batch 2`.

## 4. Single mode with an input image (image-to-video)

**Input image** (attached to the request; described in words here): a photograph at dusk. A woman in her thirties with shoulder-length auburn hair and light freckles, in a cream cable-knit sweater, sits at a small round wooden table beside a tall rain-streaked café window, holding a white ceramic mug in both hands. On the table, a saucer and a small glass vase with one white carnation; a brass pendant lamp glows above her; outside, a cool blue street with blurred amber city lights. Eye-level medium close-up, the woman centered.

**Input text:** `She sips her coffee and watches the rain.`

**Output:**

A woman in her thirties sits at a small round wooden table beside a tall rain-streaked café window, holding a white ceramic mug in both hands while the city glows softly in the dusk beyond the glass. She has shoulder-length auburn hair tucked behind one ear and light freckles across her nose, and wears a cream cable-knit sweater with the sleeves pulled down over her wrists. Initially, the woman raises the mug, blows gently across the surface, and takes a slow sip as a thin curl of steam drifts past her face. Then she lowers the mug just above the saucer and turns her gaze toward the window, watching raindrops slide down the glass in wavering lines. Throughout, she keeps this unhurried rhythm, sipping now and then, her thumb tracing the rim of the mug while her shoulders rise and fall with each calm breath. On the table, a small glass vase holds a single white carnation beside the saucer. Outside, blurred amber city lights shimmer through the streaked window as the rain keeps tapping softly against the pane. The auburn-haired woman in the cream cable-knit sweater remains the calm focal point of the table. The camera remains static, keeping the woman centered. Eye-level medium close-up. A brass pendant lamp above the table casts warm light on her face and hands, set against the cool blue glow of the rainy street. Shallow depth of field, sharp focus on her face. Quiet and reflective. Cinematic photography realistic style.

**What to notice**
- Every element of the opening (the auburn hair and freckles, the cream sweater, the mug, the vase with the carnation, the rain-streaked window, the pendant lamp) is in the image. The text only said "she"; the appearance comes from the image, and nothing the image lacks was added.
- The action picks up from the pose in the image, both hands on the mug, and becomes a repeating sip-and-watch rhythm.
- Shot size, angle, and light follow the image (eye-level medium close-up, warm lamp against the cool street). She is seated, so the camera is static and keeps her centered, as the image does.
- The image is a photograph, so the realistic style tag stays; an illustration would set a different tag.
- Without the image, only "she", the coffee, and the rain would be known: no auburn hair, sweater, or lamp could be written in.

## 5. Quick conversions

| Input fragment | Rewrite as |
|---|---|
| "drone shot orbiting a lighthouse" (stationary subject) | `The camera remains static, keeping the lighthouse centered. High-angle aerial wide shot.` |
| "zoom in on the runner" | `The camera tracks the runner, continuously keeping the runner centered.` plus a medium or wide shot |
| "camera slowly pans across a mountain range" (no clear main subject) | Balanced wide shot with `The camera remains static.`; no invented hiker |
| "zoom into her eyes" (she sits still) | Close-up of her face from the start with the static sentence; no zoom |
| "camera follows the two dancers across the stage" | `The camera tracks the two dancers, continuously keeping the two dancers centered.`; both stay in frame |
| "The dragon finally lands and folds its wings." | "Then the dragon lands and folds its wings. Throughout, it keeps slowly flexing its folded wings and exhaling thin wisps of smoke..." |
| Event 3 input: "It glides toward the diver." | "The large gray reef shark with a notched dorsal fin glides toward the scuba diver in the yellow wetsuit..." |
| "a bustling market with spice stalls, musicians, and a town crier" | Keep all three groups ("a cluster of spice merchants," "a group of street musicians," "a town crier reading from a scroll"); balanced wide shot; `The camera remains static.`; no invented protagonist |
