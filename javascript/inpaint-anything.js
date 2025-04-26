/* Inpaint‑Anything helpers patched for Gradio 4.40+                       */
/* Only selectors / MutationObserver tweaks changed – see   ⚠  markers.   */

// ‑‑‑ Utility helpers ---------------------------------------------------------
const root = () => document.querySelector("gradio-app")?.shadowRoot || document;

const waitForEl = (parent, selector, mustExist = true) =>
  new Promise(res => {
    const ok = () => !!parent.querySelector(selector) === mustExist;
    if (ok()) return res();
    const obs = new MutationObserver(() => ok() && (obs.disconnect(), res()));
    obs.observe(parent, { childList: true, subtree: true, attributes: true }); // ⚠ attr
  });

const delay = ms => new Promise(r => setTimeout(r, ms));

// ‑‑‑ Button helpers (language‑agnostic) -------------------------------------
const queryBtn = (el, rx) =>
  [...el.querySelectorAll('button[aria-label]')].find(b => rx.test(b.ariaLabel)); // ⚠ regex

const inpaintAnything_waitForElement = async (parent, selector, exist) => {
    return waitForEl(parent, selector, exist);
};

const inpaintAnything_waitForStyle = async (parent, selector, style) => {
    return new Promise((resolve) => {
        const observer = new MutationObserver(() => {
            if (!parent.querySelector(selector) || !parent.querySelector(selector).style[style]) {
                return;
            }
            observer.disconnect();
            resolve(undefined);
        });

        observer.observe(parent, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["style"],
        });

        if (!!parent.querySelector(selector) && !!parent.querySelector(selector).style[style]) {
            resolve(undefined);
        }
    });
};

const inpaintAnything_timeout = (ms) => {
    return new Promise(function (resolve, reject) {
        setTimeout(() => reject("Timeout"), ms);
    });
};

async function inpaintAnything_sendToInpaint() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, true), inpaintAnything_timeout(10000)]);

    const waitForElementToBeRemoved = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, false), inpaintAnything_timeout(10000)]);

    const updateGradioImage = async (element, url, name) => {
        const blob = await (await fetch(url)).blob();
        const file = new File([blob], name, { type: "image/png" });
        const dt = new DataTransfer();
        dt.items.add(file);

        function getClearButton() {
            let clearButton = null;
            let clearLabel = null;

            let allButtons = element.querySelectorAll("button");
            if (allButtons.length > 0) {
                for (let button of allButtons) {
                    let label = button.getAttribute("aria-label");
                    if (label && !label.includes("Edit") && !label.includes("Éditer")) {
                        clearButton = button;
                        clearLabel = label;
                        break;
                    }
                }
            }
            return [clearButton, clearLabel];
        }

        const [clearButton, clearLabel] = getClearButton();

        if (clearButton) {
            clearButton?.click();
            await waitForElementToBeRemoved(element, `button[aria-label='${clearLabel}']`);
        }

        const input = element.querySelector("input[type='file']");
        input.value = "";
        input.files = dt.files;
        input.dispatchEvent(
            new Event("change", {
                bubbles: true,
                composed: true,
            })
        );
        await waitForElementToBeInDocument(element, "button");
    };

    const inputImg = document.querySelector("#ia_input_image img");
    const maskImg = document.querySelector("#mask_out_image img");

    if (!inputImg || !maskImg) {
        return;
    }

    const inputImgDataUrl = inputImg.src;
    const maskImgDataUrl = maskImg.src;

    window.scrollTo(0, 0);
    switch_to_img2img_tab(4);

    await waitForElementToBeInDocument(document.querySelector("#img2img_inpaint_upload_tab"), "#img_inpaint_base");

    await updateGradioImage(document.querySelector("#img_inpaint_base"), inputImgDataUrl, "input.png");
    await updateGradioImage(document.querySelector("#img_inpaint_mask"), maskImgDataUrl, "mask.png");
}

async function inpaintAnything_clearSamMask() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, true), inpaintAnything_timeout(1000)]);

    const elemId = "#ia_sam_image";

    const targetElement = root().querySelector(elemId);  // ⚠ use root()
    if (!targetElement) {
        return;
    }
    
    // Reset styles
    targetElement.style.transform = null;
    targetElement.style.zIndex = null;
    targetElement.style.overflow = "auto";

    // ⚠ Find clear button using regex instead of exact text
    const clearBtn = queryBtn(targetElement, /clear/i);
    if (clearBtn) {
        clearBtn.click();
    }
    
    // ⚠ Find remove button using regex instead of exact text
    const removeBtn = queryBtn(targetElement, /remove/i);
    if (!removeBtn) {
        return;
    }
    
    if (typeof inpaintAnything_clearSamMask.clickRemoveImage === "undefined") {
        inpaintAnything_clearSamMask.clickRemoveImage = () => {
            targetElement.style.transform = null;
            targetElement.style.zIndex = null;
        };
    } else {
        removeBtn.removeEventListener("click", inpaintAnything_clearSamMask.clickRemoveImage);
    }
    removeBtn.addEventListener("click", inpaintAnything_clearSamMask.clickRemoveImage);
}

async function inpaintAnything_clearSelMask() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, true), inpaintAnything_timeout(1000)]);

    const elemId = "#ia_sel_mask";

    const targetElement = root().querySelector(elemId);  // ⚠ use root()
    if (!targetElement) {
        return;
    }

    // Reset styles
    targetElement.style.transform = null;
    targetElement.style.zIndex = null;
    targetElement.style.overflow = "auto";

    // ⚠ Find clear button using regex instead of exact text
    const clearBtn = queryBtn(targetElement, /clear/i);
    if (clearBtn) {
        clearBtn.click();
    }
    
    // ⚠ Find remove button using regex instead of exact text
    const removeBtn = queryBtn(targetElement, /remove/i);
    if (!removeBtn) {
        return;
    }

    if (typeof inpaintAnything_clearSelMask.clickRemoveImage === "undefined") {
        inpaintAnything_clearSelMask.clickRemoveImage = () => {
            targetElement.style.transform = null;
            targetElement.style.zIndex = null;
        };
    } else {
        removeBtn.removeEventListener("click", inpaintAnything_clearSelMask.clickRemoveImage);
    }
    removeBtn.addEventListener("click", inpaintAnything_clearSelMask.clickRemoveImage);
}

async function inpaintAnything_initSamSelMask() {
    inpaintAnything_clearSamMask();
    inpaintAnything_clearSelMask();
}

async function inpaintAnything_getPrompt(tabName, promptId, negPromptId) {
    const tabTxt2img = document.querySelector(`#tab_${tabName}`);
    if (!tabTxt2img) {
        return;
    }

    const txt2imgPrompt = tabTxt2img.querySelector(`#${tabName}_prompt textarea`);
    const txt2imgNegPrompt = tabTxt2img.querySelector(`#${tabName}_neg_prompt textarea`);
    if (!txt2imgPrompt || !txt2imgNegPrompt) {
        return;
    }

    const iaSdPrompt = document.querySelector(`#${promptId} textarea`);
    const iaSdNPrompt = document.querySelector(`#${negPromptId} textarea`);
    if (!iaSdPrompt || !iaSdNPrompt) {
        return;
    }

    iaSdPrompt.value = txt2imgPrompt.value;
    iaSdNPrompt.value = txt2imgNegPrompt.value;

    iaSdPrompt.dispatchEvent(new Event("input", { bubbles: true }));
    iaSdNPrompt.dispatchEvent(new Event("input", { bubbles: true }));
}

async function inpaintAnything_getTxt2imgPrompt() {
    inpaintAnything_getPrompt("txt2img", "ia_sd_prompt", "ia_sd_n_prompt");
}

async function inpaintAnything_getImg2imgPrompt() {
    inpaintAnything_getPrompt("img2img", "ia_sd_prompt", "ia_sd_n_prompt");
}

async function inpaintAnything_webuiGetTxt2imgPrompt() {
    inpaintAnything_getPrompt("txt2img", "ia_webui_sd_prompt", "ia_webui_sd_n_prompt");
}

async function inpaintAnything_webuiGetImg2imgPrompt() {
    inpaintAnything_getPrompt("img2img", "ia_webui_sd_prompt", "ia_webui_sd_n_prompt");
}

async function inpaintAnything_cnGetTxt2imgPrompt() {
    inpaintAnything_getPrompt("txt2img", "ia_cn_sd_prompt", "ia_cn_sd_n_prompt");
}

async function inpaintAnything_cnGetImg2imgPrompt() {
    inpaintAnything_getPrompt("img2img", "ia_cn_sd_prompt", "ia_cn_sd_n_prompt");
}

onUiLoaded(async () => {
    const elementIDs = {
        ia_sam_image: "#ia_sam_image",
        ia_sel_mask: "#ia_sel_mask",
        ia_out_image: "#ia_out_image",
        ia_cleaner_out_image: "#ia_cleaner_out_image",
        ia_webui_out_image: "#ia_webui_out_image",
        ia_cn_out_image: "#ia_cn_out_image"
    };

    function setStyleHeight(elemId, height) {
        const elem = gradioApp().querySelector(elemId);
        if (elem) {
            if (!elem.style.height) {
                elem.style.height = height;
                const observer = new MutationObserver(() => {
                    const divPreview = elem.querySelector(".preview");
                    if (divPreview) {
                        divPreview.classList.remove("fixed-height");
                    }
                });
                observer.observe(elem, {
                    childList: true,
                    attributes: true,
                    attributeFilter: ["class"],
                });
            }
        }
    }

    setStyleHeight(elementIDs.ia_out_image, "520px");
    setStyleHeight(elementIDs.ia_cleaner_out_image, "520px");
    setStyleHeight(elementIDs.ia_webui_out_image, "520px");
    setStyleHeight(elementIDs.ia_cn_out_image, "520px");

    // Default config
    //const defaultHotkeysConfig = {
    //    canvas_hotkey_reset: "KeyR",
    //    canvas_hotkey_fullscreen: "KeyS",
    //};
    const HOTKEY_RESET = "r";
    const HOTKEY_FULLSCREEN = "s";

    const elemData = {};
    let activeElement;

    function applyZoomAndPan(elemId) {
        const targetElement = gradioApp().querySelector(elemId);

        if (!targetElement) {
            console.log("Element not found");
            return;
        }

        targetElement.style.transformOrigin = "0 0";

        elemData[elemId] = {
            zoomLevel: 1,
            panX: 0,
            panY: 0,
        };
        let fullScreenMode = false;

        // Toggle the zIndex of the target element between two values, allowing it to overlap or be overlapped by other elements
        function toggleOverlap(forced = "") {
            // const zIndex1 = "0";
            const zIndex1 = null;
            const zIndex2 = "998";

            targetElement.style.zIndex = targetElement.style.zIndex !== zIndex2 ? zIndex2 : zIndex1;

            if (forced === "off") {
                targetElement.style.zIndex = zIndex1;
            } else if (forced === "on") {
                targetElement.style.zIndex = zIndex2;
            }
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        function fitToElement() {
            //Reset Zoom
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const parentElement = targetElement.parentElement;
            const screenWidth = parentElement.clientWidth;
            const screenHeight = parentElement.clientHeight;

            // Get element's coordinates relative to the parent element
            const elementRect = targetElement.getBoundingClientRect();
            const parentRect = parentElement.getBoundingClientRect();
            const elementX = elementRect.x - parentRect.x;

            // Calculate scale and offsets
            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            const transformOrigin = window.getComputedStyle(targetElement).transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            const offsetX = (screenWidth - elementWidth * scale) / 2 - originXValue * (1 - scale);
            const offsetY = (screenHeight - elementHeight * scale) / 2.5 - originYValue * (1 - scale);

            // Apply scale and offsets to the element
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            // Update global variables
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = false;
            toggleOverlap("off");
        }

        // Reset the zoom level and pan position of the target element to their initial values
        function resetZoom() {
            elemData[elemId] = {
                zoomLevel: 1,
                panX: 0,
                panY: 0,
            };

            // fixCanvas();
            targetElement.style.transform = `scale(${elemData[elemId].zoomLevel}) translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px)`;

            // const canvas = gradioApp().querySelector(`${elemId} canvas[key="interface"]`);

            toggleOverlap("off");
            fullScreenMode = false;

            // if (
            //     canvas &&
            //     parseFloat(canvas.style.width) > 865 &&
            //     parseFloat(targetElement.style.width) > 865
            // ) {
            //     fitToElement();
            //     return;
            // }

            // targetElement.style.width = "";
            // if (canvas) {
            //     targetElement.style.height = canvas.style.height;
            // }
            targetElement.style.width = null;
            targetElement.style.height = (CANVAS_H + 2*PAD) || (749 + 40) + "px"; // ⚠ Match height from Python code
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        // Fullscreen mode
        function fitToScreen() {
            const canvas = gradioApp().querySelector(`${elemId} canvas[key="interface"]`);
            const img = gradioApp().querySelector(`${elemId} img`);

            if (!canvas && !img) return;

            // if (canvas.offsetWidth > 862) {
            //     targetElement.style.width = canvas.offsetWidth + "px";
            // }

            if (fullScreenMode) {
                resetZoom();
                fullScreenMode = false;
                return;
            }

            //Reset Zoom
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            // Get scrollbar width to right-align the image
            const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const screenWidth = window.innerWidth - scrollbarWidth;
            const screenHeight = window.innerHeight;

            // Get element's coordinates relative to the page
            const elementRect = targetElement.getBoundingClientRect();
            const elementY = elementRect.y;
            const elementX = elementRect.x;

            // Calculate scale and offsets
            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            // Get the current transformOrigin
            const computedStyle = window.getComputedStyle(targetElement);
            const transformOrigin = computedStyle.transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            // Calculate offsets with respect to the transformOrigin
            const offsetX = (screenWidth - elementWidth * scale) / 2 - elementX - originXValue * (1 - scale);
            const offsetY = (screenHeight - elementHeight * scale) / 2 - elementY - originYValue * (1 - scale);

            // Apply scale and offsets to the element
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            // Update global variables
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = true;
            toggleOverlap("on");
        }

        // Reset zoom when uploading a new image (updated for Gradio 4)
        const fileInput = gradioApp().querySelector(`${elemId} input[type="file"]`);
        if (fileInput) {
            fileInput.addEventListener("click", resetZoom);
        }

        // Handle keydown events
        function handleKeyDown(event) {
            // Disable key locks to make pasting from the buffer work correctly
            if (
                (event.ctrlKey && event.code === "KeyV") ||
                (event.ctrlKey && event.code === "KeyC") ||
                event.code === "F5"
            ) {
                return;
            }

            // before activating shortcut, ensure user is not actively typing in an input field
            if (event.target.nodeName === "TEXTAREA" || event.target.nodeName === "INPUT") {
                return;
            }

            //const hotkeyActions = {
            //    [defaultHotkeysConfig.canvas_hotkey_reset]: resetZoom,
            //    [defaultHotkeysConfig.canvas_hotkey_fullscreen]: fitToScreen,
            //};
            const k = event.key.toLowerCase();
            let action = null;
            if (k === HOTKEY_RESET)        action = resetZoom;
            else if (k === HOTKEY_FULLSCREEN) action = fitToScreen;
            //const action = hotkeyActions[event.code];
            
            if (action) {
                event.preventDefault();
                action(event);
            }
        }

        // Handle events only inside the targetElement
        let isKeyDownHandlerAttached = false;

        function handleMouseMove() {
            if (!isKeyDownHandlerAttached) {
                document.addEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = true;

                activeElement = elemId;
            }
        }

        function handleMouseLeave() {
            if (isKeyDownHandlerAttached) {
                document.removeEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = false;

                activeElement = null;
            }
        }

        // Add mouse event handlers
        targetElement.addEventListener("mousemove", handleMouseMove);
        targetElement.addEventListener("mouseleave", handleMouseLeave);
    }

    applyZoomAndPan(elementIDs.ia_sam_image);
    applyZoomAndPan(elementIDs.ia_sel_mask);
    // applyZoomAndPan(elementIDs.ia_out_image);
    // applyZoomAndPan(elementIDs.ia_cleaner_out_image);
    // applyZoomAndPan(elementIDs.ia_webui_out_image);
    // applyZoomAndPan(elementIDs.ia_cn_out_image);
});
