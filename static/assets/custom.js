document.addEventListener("DOMContentLoaded", function () {


 



    //  иконки для AG GRID
    window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

    window.dashAgGridComponentFunctions.officeIconRenderer = function (props) {
        const value = props.value;
        if (value === "Офис") {
            return "🏢 " + value;
        } else if (value === "Торговля") {
            return "🛍️ " + value;
        } else if (value === "Парковка") {
            return "🚗 " + value;
        } else if (value === "Склад") {
            return "📦 " + value;
        }
        else if (value === "Спецпроект") {
            return "🧩 " + value;
        } else if (value === "Телекомуникации") {
            return "📡 " + value;
        } else if (value === "Реклама") {
            return "📢 " + value;
        }
        return value;
    };






    // Анимация логотипа
    const logo = document.getElementById("logo-img");
    if (logo) {
        logo.addEventListener("click", function () {
            logo.classList.add("rotating");
            setTimeout(() => logo.classList.remove("rotating"), 600);
        });
    }

    // Кнопка скачивания графика
    // Запускаем сразу
console.log("✅ custom.js подключен!");

// Пытаемся найти кнопки и контейнеры с интервалом
const intervalId = setInterval(function() {
    const button1 = document.getElementById('download_chart_btn');
    const chart1 = document.getElementById('rra_area_chart_container');

    const button2 = document.getElementById('download_button_wf');
    const chart2 = document.getElementById('rrca_wf_chart');

    // Проверяем, что все элементы найдены
    if (button1 && chart1 && button2 && chart2) {
        console.log("✅ Обе кнопки и оба контейнера найдены!");

        // Обработчик для первой кнопки
        button1.addEventListener('click', function() {
            console.log("Клик по кнопке 1!");

            html2canvas(chart1, {
                backgroundColor: "#ffffff",
                scale: 5
            }).then(canvas => {
                const link = document.createElement('a');
                link.download = 'Анализ арендного портфеля.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            });
        });

        // Обработчик для второй кнопки
        button2.addEventListener('click', function() {
            console.log("Клик по кнопке 2!");

            html2canvas(chart2, {
                backgroundColor: "#ffffff",
                scale: 5
            }).then(canvas => {
                const link = document.createElement('a');
                link.download = 'Анализ изменения арендного портфеля.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            });
        });

        // Останавливаем проверку
        clearInterval(intervalId);
    }
}, 500); // Проверяем каждые 500 мс




    // dmc функции
    var dmcfuncs = window.dashMantineFunctions = window.dashMantineFunctions || {};

    dmcfuncs.formatNumberIntl = function(value) {
    if (value == null || isNaN(value)) return "₽0";
    const absValue = Math.abs(value);
    let formatted;
    if (absValue >= 1_000_000) {
        formatted = "₽" + (absValue / 1_000_000).toFixed(1).replace(".", ",") + " млн";
    } else {
        formatted = "₽" + (absValue / 1_000).toFixed(0).replace(".", ",") + " тыс";
    }
    if (value < 0) {
        formatted = "-" + formatted;
    }
    return formatted;
};


//     dmcfuncs.formatQuant = function(value) {
//     if (value == null || isNaN(value)) return "0 м²";
//     const absValue = Math.abs(value);
//     let formatted;
//     if (absValue >= 1000) {
//         formatted = (absValue / 1000).toFixed(1).replace(".", ",") + " тыс м²";
//     } else {
//         formatted = absValue.toFixed(0) + " м²";
//     }
//     if (value < 0) {
//         formatted = "-" + formatted;
//     }
//     return formatted;
// };


    dmcfuncs.formatQuant = function (value) {
    const NNBSP = '\u202F'; // узкий неразрывный пробел
    if (value == null || isNaN(value)) return '0' + NNBSP + 'м²';

    const abs = Math.abs(value);
    const sign = value < 0 ? '−' : ''; // типографский минус

    const nf0 = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 });
    const nf1 = new Intl.NumberFormat('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 1 });

    if (abs >= 1000) {
        // пример: 1,2 тыс м²
        return sign + nf1.format(abs / 1000) + NNBSP + 'тыс' + NNBSP + 'м²';
    } else {
        // пример: 950 м² (внутри числа разделитель тысяч — неразрывный пробел)
        return sign + nf0.format(abs) + NNBSP + 'м²';
    }
    };



    dmcfuncs.formatPercentIntl = function(value) {
        if (value == null || isNaN(value)) return "0 %";
        if (Math.abs(value) < 0.1) {
            return "<0,1 %";
        }
        return value.toFixed(1).replace(".", ",") + " %";
    };

    dmcfuncs.formatIntl = function(value) {
        if (value == null || isNaN(value)) return "0";
        if (Math.abs(value) < 0.1) {
            return "0";
        }
        return value.toFixed(0).replace(".", ",") + " шт";
    };

    window.dashMantineFunctions.formatRubles = (value) => {
        return '₽' + (value / 1000000).toFixed(1) + ' млн';
    };

    dmcfuncs.formatRublesMillions = function(value) {
        console.log("Получено значение:", value);
        if (value == null || isNaN(value)) return "₽0 млн";
        return "₽" + (value / 1_000_000).toFixed(1).replace(".", ",") + " млн";
    };
    // console.log("Функция formatRublesMillions загружена:", dmcfuncs.formatRublesMillions);

    dmcfuncs.formatMonthLabel = function(value, { monthDict }) {
        return monthDict[value] || value;
    };


    dmcfuncs.formatARPU = function(value) {
    if (value == null || isNaN(value)) return "₽0";
    const absValue = Math.abs(value);
    let formatted;
    if (absValue >= 1_000_000) {
        formatted = "₽" + (absValue / 1_000_000)
            .toFixed(1)
            .replace(".", ",") + " млн / м²";
    } else {
        formatted = "₽" + absValue
            .toFixed(0)
            .toString()
            .replace(/\B(?=(\d{3})+(?!\d))/g, " ");  // добавляем пробелы для тысяч
    }
    if (value < 0) {
        formatted = "-" + formatted;
    }
    return formatted;
};

    

//     dmcfuncs.formatARPU = function(value) {
//     if (value == null || isNaN(value)) return "₽0";
//     const absValue = Math.abs(value);
//     let formatted;
//     if (absValue >= 1_000_000) {
//         formatted = "₽" + (absValue / 1_000_000).toFixed(1).replace(".", ",") + " млн / м2";
//     } else {
//         formatted = "₽" + (absValue).toFixed(0).replace(".", ",");
//     }
//     if (value < 0) {
//         formatted = "-" + formatted;
//     }
//     return formatted;
// };
});


// === Dash AG Grid functions (ГЛОБАЛЬНО) ===
window.dashAgGridFunctions = window.dashAgGridFunctions || {};

// форматтер для колонки "Осталось_дней"
window.dashAgGridFunctions.daysFormatter = function (params) {
  var isTotal = (params && params.node && params.node.footer) ||
                (params && params.data && params.data.Арендатор === 'ИТОГО');
  var v = Number(params && params.value);
  if (isTotal || !isFinite(v)) return '';
  return Math.round(v) + ' дн.';
};

// правила раскраски (без optional chaining)
window.dashAgGridFunctions.daysIsRed = function (params) {
  var isTotal = (params && params.node && params.node.footer) ||
                (params && params.data && params.data.Арендатор === 'ИТОГО');
  if (isTotal) return false;
  var v = Number(params && params.value);
  return isFinite(v) && v <= 30;
};
window.dashAgGridFunctions.daysIsOrange = function (params) {
  var isTotal = (params && params.node && params.node.footer) ||
                (params && params.data && params.data.Арендатор === 'ИТОГО');
  if (isTotal) return false;
  var v = Number(params && params.value);
  return isFinite(v) && v > 30 && v <= 90;
};
window.dashAgGridFunctions.daysIsGreen = function (params) {
  var isTotal = (params && params.node && params.node.footer) ||
                (params && params.data && params.data.Арендатор === 'ИТОГО');
  if (isTotal) return false;
  var v = Number(params && params.value);
  return isFinite(v) && v > 90;
};
